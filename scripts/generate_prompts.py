import json
from jinja2 import Environment, FileSystemLoader

def generate_prompts(personas_path, instruments_path, template_path, output_path):
    """Generates prompts for each persona and experimental condition."""
    with open(personas_path, 'r') as f:
        personas = json.load(f)['personas']
    with open(instruments_path, 'r') as f:
        instruments = json.load(f)

    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template(template_path)

    all_prompts = {}

    for persona in personas:
        persona_id_str = f"persona_{persona['id']:03d}"
        all_prompts[persona_id_str] = {
            'condition_1': {},
            'condition_2': {},
            'condition_3': {}
        }

        # Condition 1: Clear context after each question
        for scale_name, scale_data in instruments.items():
            if scale_name not in all_prompts[persona_id_str]['condition_1']:
                all_prompts[persona_id_str]['condition_1'][scale_name] = {}
            for subscale_name, questions_data in scale_data.items():
                if subscale_name == 'scale_id' or subscale_name == 'response_scale':
                    continue
                if subscale_name not in all_prompts[persona_id_str]['condition_1'][scale_name]:
                    all_prompts[persona_id_str]['condition_1'][scale_name][subscale_name] = {}
                for question_id, question_text in questions_data.items():
                    prompt = template.render(persona=persona, question=question_text)
                    all_prompts[persona_id_str]['condition_1'][scale_name][subscale_name][question_id] = prompt

        # Condition 2: Clear context after each scale
        for scale_name, scale_data in instruments.items():
            if scale_name == 'scale_id' or scale_name == 'response_scale':
                continue
            # For condition 2, we need to pass the entire scale data to the template
            # This assumes the template can iterate through questions within a scale
            prompt = template.render(persona=persona, scale_data=scale_data)
            all_prompts[persona_id_str]['condition_2'][scale_name] = prompt

        # Condition 3: Maintain full context, randomizing order of scales
        # The randomization will happen in the simulation script, so here we just pass all instruments
        prompt = template.render(persona=persona, instruments=instruments)
        all_prompts[persona_id_str]['condition_3'] = prompt

    with open(output_path, 'w') as f:
        json.dump(all_prompts, f, indent=2)

if __name__ == '__main__':
    generate_prompts(
        'outputs/personas.json',
        'outputs/instruments.json',
        'prompts/system_prompt_template.jinja',
        'outputs/prompts.json'
    )
