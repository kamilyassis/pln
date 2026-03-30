from agno.agent import Agent
from agno.models.openai import OpenAIChat
from pydantic import BaseModel, Field
from typing import List
import asyncio
from tqdm.asyncio import tqdm_asyncio
from dotenv import load_dotenv
import os
import csv

load_dotenv()

from enum import Enum

class Label(Enum):
    LOADED_LANGUAGE = "Loaded_Language"
    NAME_CALLING_LABELING = "Name_Calling-Labeling"
    DOUBT = "Doubt"
    REPETITION = "Repetition"
    APPEAL_TO_FEAR_PREJUDICE = "Appeal_to_Fear-Prejudice"
    FLAG_WAVING = "Flag_Waving"
    EXAGGERATION_MINIMISATION = "Exaggeration-Minimisation"
    NO_LABEL = "No_Label"
    ERROR = "ERROR"


class SentenceLabel(BaseModel):
    label: Label = Field()
    justification: str = Field()


def create_labeler(
    model_name: str = "gpt-4.1-nano",
    temperature: float = 0.1,
    max_tokens: int = 150,
    prompt_path: str = "prompt/semeval-label-prompt.txt",
    output_model: BaseModel = SentenceLabel
) -> Agent:
    """Creates and returns a configured labeling agent."""
    with open(prompt_path, "r") as f:
        system_prompt = f.read()
    
    model = OpenAIChat(
        id=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    return Agent(
        model=model,
        instructions=system_prompt,
        markdown=False,
        response_model=output_model,
    )

def label_sentence(agent: Agent, sentence: str) -> SentenceLabel:
    """Processes a sentence and returns structured label + justification."""
    try:
        response = agent.run(sentence, stream=False)
        
        # Check if response.content is already a SentenceLabel
        if isinstance(response.content, SentenceLabel):
            return response.content
        
        # If it's a string, try to parse it as JSON
        if isinstance(response.content, str):
            import json
            try:
                data = json.loads(response.content)
                # Map label string to Label enum
                label_str = data.get('label', 'ERROR').upper().replace('-', '_').replace(' ', '_')
                return SentenceLabel(
                    label=Label[label_str],
                    justification=data.get('justification', 'No justification provided')
                )
            except (json.JSONDecodeError, KeyError):
                # If JSON parsing fails, return as ERROR
                return SentenceLabel(label=Label.ERROR, justification=f"Failed to parse: {response.content}")
        
        return SentenceLabel(label=Label.ERROR, justification=str(response.content))
        
    except Exception as e:
        import traceback
        print(f"Error processing: {e}")
        traceback.print_exc()
        return SentenceLabel(label=Label.ERROR, justification=str(e))

async def save_labels_to_csv(
    sentences: List[str],
    labels: List[SentenceLabel],
    output_file: str = "labeled_sentences.csv"
):
    file_exists = os.path.exists(output_file)
    
    with open(output_file, 'a' if file_exists else 'w', newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow(['sentence', 'label', 'justification'])
        
        # Write data rows
        for sentence, label in zip(sentences, labels):
            writer.writerow([sentence, label.label.value, label.justification])

async def process_label_batch(
    sentences: List[str],
    agent: Agent,
    batch_size: int = 10,
    output_file: str = "labeled_sentences.csv"
) -> List[SentenceLabel]:
    """Processes sentences in concurrent batches."""
    all_labels = []
    
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        batch_results = await tqdm_asyncio.gather(
            *[asyncio.to_thread(label_sentence, agent, s) for s in batch],
            desc=f"Processing batch {i//batch_size + 1}"
        )
        all_labels.extend(batch_results)
        await save_labels_to_csv(batch, batch_results, output_file)
    
    return all_labels

if __name__ == "__main__":
    import pandas as pd
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", help="Input CSV file")
    parser.add_argument("-o", "--output_file", help="Output CSV file")
    parser.add_argument("-m", "--model", help="Model name")
    parser.add_argument("-b", "--batch_size", type=int, help="Batch size")
    parser.add_argument("-p", "--prompt_path", help="Path to the prompt file")
    
    args = parser.parse_args()
    
    sentences = []
    df = pd.read_csv(args.input_file, sep='|')

    if 'text' in df.columns:
        sentences = df['text'].tolist()
    elif 'sentence' in df.columns:
        sentences = df['sentence'].tolist()
    else:
        print("Error: No 'text' or 'sentence' column found in the input CSV.")

    df_out = pd.read_csv(args.output_file) if os.path.exists(args.output_file) else pd.DataFrame(columns=['sentence', 'label', 'justification'])

    # diff sentences from df out and df in
    to_process = df[~df['text'].isin(df_out['sentence'])]
    to_process = pd.concat([to_process, df_out[df_out['label'] == 'ERROR']], ignore_index=True)
    sentences = to_process['text'].drop_duplicates().tolist()

    print(f"Total sentences: {len(df)}")
    print(f"Sentences to process: {len(sentences)}")

    print(f"Batches to process: {len(sentences) // args.batch_size + 1}")

    agent = create_labeler(
        model_name=args.model,
        temperature=0.1,
        max_tokens=150,
        prompt_path=args.prompt_path
    )

    asyncio.run(process_label_batch(sentences, agent, batch_size=args.batch_size, output_file=args.output_file))