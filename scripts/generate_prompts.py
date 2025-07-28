import json
from jinja2 import Environment, FileSystemLoader

def generate_prompts(personas_path, instruments_path, template_path):
    """Generates prompts for each persona and experimental condition."""
    with open(personas_path, 'r') as f:
        personas = json.load(f)['personas']
    with open(instruments_path, 'r') as f:
        instruments = json.load(f)

    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template(template_path)

    prompts = {
        'condition_1': [],
        'condition_2': [],
        'condition_3': []
    }

    for persona in personas:
        # Condition 1: Clear context after each question
        for scale, subscales in instruments.items():
            for subscale, questions in subscales.items():
                if subscale == 'scale_id' or subscale == 'response_scale':
                    continue
                for question_id, question_text in questions.items():
                    prompt = template.render(persona=persona, question=question_text)
                    prompts['condition_1'].append(prompt)

        # Condition 2: Clear context after each scale
        for scale, subscales in instruments.items():
            prompt = template.render(persona=persona, questions=subscales)
            prompts['condition_2'].append(prompt)

        # Condition 3: Maintain full context, randomizing order of scales
        # This will be handled in the simulation script
        prompt = template.render(persona=persona, instruments=instruments)
        prompts['condition_3'].append(prompt)

    with open('outputs/prompts.json', 'w') as f:
        json.dump(prompts, f, indent=2)

if __name__ == '__main__':
    generate_prompts(
        'outputs/personas.json',
        'outputs/instruments.json',
        'prompts/system_prompt_template.jinja'
    )
