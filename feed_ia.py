import os
from mlx_lm import load
from mlx_lm.generate import generate_step
import mlx.core as mx

model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

directory_path = "txts"
contextos = []

for filename in os.listdir(directory_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            contextos.append((filename, content))

print(f"{len(contextos)} contextos carregados.")

def gerar_resposta(prompt, max_tokens=256):
    input_tokens = mx.array(tokenizer.encode(prompt))
    output = ""
    for token, _ in generate_step(input_tokens, model, max_tokens=max_tokens):
        output += tokenizer.decode([token])
    return output.strip()

while True:
    pergunta = input("\nPergunta (ou 'sair' para encerrar): ").strip()
    if pergunta.lower() == "sair":
        break
    if not pergunta:
        print("Digite uma pergunta válida.")
        continue

    prompt_escolha = "Contextos:\n\n"
    for idx, (_, contexto) in enumerate(contextos, start=1):
        prompt_escolha += f"{idx}. {contexto[:500]}...\n\n"  # corta para não ficar muito grande

    prompt_escolha += f"Com base nesses contextos, qual deles responde melhor à pergunta abaixo?\nPergunta: {pergunta}\nResponda apenas com o número do contexto mais relevante."

    resposta_modelo = gerar_resposta(prompt_escolha, max_tokens=10)

    try:
        contexto_idx = int("".join(filter(str.isdigit, resposta_modelo)))
        if not (1 <= contexto_idx <= len(contextos)):
            raise ValueError
    except ValueError:
        print(" Não foi possível identificar o melhor contexto. Tente novamente.")
        continue

    contexto_escolhido = contextos[contexto_idx - 1][1]

    prompt_final = f"{contexto_escolhido}\n\nPergunta: {pergunta}"
    resposta_final = gerar_resposta(prompt_final, max_tokens=256)

    print("\n Resposta:")
    print(resposta_final)
    print("-" * 50)
