from mlx_lm import load
from mlx_lm.generate import generate_step
import mlx.core as mx

# Carrega o modelo e o tokenizer
model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

print("✅ Modelo carregado com sucesso! Digite sua pergunta.\n")

while True:
    pergunta = input("🧠 Pergunta (ou digite 'sair' para encerrar): ").strip()
    
    if pergunta.lower() == "sair":
        print("Encerrando...")
        break
    
    if not pergunta:
        print("Por favor, digite uma pergunta válida.")
        continue

    # Codifica a pergunta como prompt (sem contexto adicional)
    input_tokens = mx.array(tokenizer.encode(pergunta))

    # Gera a resposta
    output = ""
    for token, _ in generate_step(input_tokens, model, max_tokens=256):
        output += tokenizer.decode([token])

    print("\n🤖 Resposta:")
    print(output)
    print("-" * 50)
