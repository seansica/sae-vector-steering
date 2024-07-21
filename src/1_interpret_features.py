import os
from openai import OpenAI
from typing_extensions import override
from openai import AssistantEventHandler
from dotenv import load_dotenv


def analyze_sae_features(json_file_path, openai_api_key):
    client = OpenAI(api_key=openai_api_key)

    # Step 1: Create a new Assistant with File Search Enabled
    assistant = client.beta.assistants.create(
        name="SAE Feature Analyzer",
        instructions="You are an expert in analyzing sparse autoencoder features. Use the provided JSON file to interpret and explain the features.",
        model="gpt-4o",
        tools=[{"type": "file_search"}],
    )

    # Step 2: Upload files and add them to a Vector Store
    vector_store = client.beta.vector_stores.create(name="SAE Features")

    with open(json_file_path, "rb") as file:
        file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id, files=[file]
        )

    print(f"File batch status: {file_batch.status}")
    print(f"File counts: {file_batch.file_counts}")

    # Step 3: Update the assistant to use the new Vector Store
    assistant = client.beta.assistants.update(
        assistant_id=assistant.id,
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
    )

    # Step 4: Create a thread
    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": "Please analyze and interpret the features from the SAE JSON file. Find all features related to Pokemon, including Pokemon names you are familiar with.",
            }
        ]
    )

    # Step 5: Create a run and check the output
    class EventHandler(AssistantEventHandler):
        @override
        def on_text_created(self, text) -> None:
            print(f"\nassistant > ", end="", flush=True)

        @override
        def on_tool_call_created(self, tool_call):
            print(f"\nassistant > {tool_call.type}\n", flush=True)

        @override
        def on_message_done(self, message) -> None:
            message_content = message.content[0].text
            annotations = message_content.annotations
            citations = []
            for index, annotation in enumerate(annotations):
                message_content.value = message_content.value.replace(
                    annotation.text, f"[{index}]"
                )
                if file_citation := getattr(annotation, "file_citation", None):
                    cited_file = client.files.retrieve(file_citation.file_id)
                    citations.append(f"[{index}] {cited_file.filename}")

            print(message_content.value)
            print("\n".join(citations))

    with client.beta.threads.runs.stream(
        thread_id=thread.id,
        assistant_id=assistant.id,
        event_handler=EventHandler(),
    ) as stream:
        stream.until_done()


# Usage
if __name__ == "__main__":
    load_dotenv()
    json_file_path = "top_activated_words.json"
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    analyze_sae_features(json_file_path, openai_api_key)
