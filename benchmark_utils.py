import random
random.seed(0)

def process_prompt(prefix, prompt, suffix):
    return prefix + prompt + suffix

def read_prompts(num_prompts=30):
    prompts = []
    with open("/home/slida/DF-Serving/PartiPrompts_Detail_eval/PartiPrompts_Detail.tsv", 'r') as fr:
        for line in fr:
            parts = line.strip().split("\t")
            assert len(parts) >= 3, parts
            prompts.append(parts[0])
        random.shuffle(prompts)
    
    return prompts[:num_prompts]

if __name__ == "__main__":
    prompts = read_prompts(265)
    for i, prompt in enumerate(prompts):
        print(f"Prompt {i+1}: {prompt}")