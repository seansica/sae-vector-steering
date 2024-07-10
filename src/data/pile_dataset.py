import h5py
from tqdm.notebook import tqdm
from torch.utils.data import IterableDataset
from collections import defaultdict


class PileDataset(IterableDataset):
    def __init__(self, dataset_iterator, context_length, tokenizer):
        self.dataset_iterator = dataset_iterator
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.current_tokens = []
        self.current_pile_set_name = None
        self.pile_set_distribution = defaultdict(int)
        self.total_tokens = 0

    def __iter__(self):
        return self

    def __next__(self):
        while len(self.current_tokens) < self.context_length:
            try:
                example = next(self.dataset_iterator)
                text = example["text"] + "<|endoftext|>"
                new_tokens = self.tokenizer.encode(
                    text, allowed_special={"<|endoftext|>"}
                )
                self.current_tokens.extend(new_tokens)
                self.current_pile_set_name = example["meta"]["pile_set_name"]
                self.pile_set_distribution[self.current_pile_set_name] += len(
                    new_tokens
                )
                self.total_tokens += len(new_tokens)
            except StopIteration:
                if not self.current_tokens:
                    raise StopIteration
                break
            except Exception as e:
                print(f"Error processing example: {e}")
                continue

        if len(self.current_tokens) >= self.context_length:
            tokens = self.current_tokens[: self.context_length]
            self.current_tokens = self.current_tokens[self.context_length :]
        else:
            tokens = self.current_tokens
            self.current_tokens = []

        return tokens, self.current_pile_set_name

    def print_distribution(self):
        print("Data Distribution:")
        for pile_set, count in self.pile_set_distribution.items():
            print(f"{pile_set}: {count} tokens ({count/self.total_tokens*100:.2f}%)")

    def export_to_h5(self, output_file, chunk_size=1000000, max_tokens=None):
        with h5py.File(output_file, "w") as f:
            dset = f.create_dataset(
                "tokens", shape=(0,), maxshape=(None,), dtype="i4", chunks=True
            )

            total_tokens = 0
            buffer = []
            samples_processed = 0

            pbar = tqdm(
                total=max_tokens or float("inf"), desc="Processing tokens", unit="tok"
            )

            try:
                for tokens, pile_set_name in self:
                    buffer.extend(tokens)

                    if len(buffer) >= chunk_size:
                        dset.resize(total_tokens + len(buffer), axis=0)
                        dset[total_tokens:] = buffer
                        total_tokens += len(buffer)
                        pbar.update(len(buffer))
                        buffer = []

                    samples_processed += 1

                    if max_tokens and total_tokens >= max_tokens:
                        print(f"Reached max tokens: {total_tokens}")
                        break

                    if samples_processed % 1000 == 0:
                        print(
                            f"Processed {samples_processed} samples, {total_tokens} tokens"
                        )

            except Exception as e:
                print(f"Error in main loop: {e}")

            if buffer:
                dset.resize(total_tokens + len(buffer), axis=0)
                dset[total_tokens:] = buffer
                total_tokens += len(buffer)
                pbar.update(len(buffer))

            pbar.close()

            # Save the distribution to the HDF5 file
            distribution_group = f.create_group("distribution")
            for pile_set, count in self.pile_set_distribution.items():
                distribution_group.attrs[pile_set] = count

        print(f"Tokenization complete. Total tokens: {total_tokens}")
        print(f"Samples processed: {samples_processed}")
        return output_file

    def split(self, train_ratio, val_ratio, test_ratio):
        assert (
            abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        ), "Ratios must sum to 1"

        train_tokens = int(self.total_tokens * train_ratio)
        val_tokens = int(self.total_tokens * val_ratio)
        test_tokens = self.total_tokens - train_tokens - val_tokens

        train_dataset = PileDataset(
            self.dataset_iterator, self.context_length, self.tokenizer
        )
        val_dataset = PileDataset(
            self.dataset_iterator, self.context_length, self.tokenizer
        )
        test_dataset = PileDataset(
            self.dataset_iterator, self.context_length, self.tokenizer
        )

        current_tokens = 0
        current_dataset = train_dataset

        for tokens, pile_set_name in self:
            if current_tokens < train_tokens:
                current_dataset = train_dataset
            elif current_tokens < train_tokens + val_tokens:
                current_dataset = val_dataset
            else:
                current_dataset = test_dataset

            current_dataset.current_tokens.extend(tokens)
            current_dataset.pile_set_distribution[pile_set_name] += len(tokens)
            current_dataset.total_tokens += len(tokens)

            current_tokens += len(tokens)

            if current_tokens >= self.total_tokens:
                break

        return train_dataset, val_dataset, test_dataset
