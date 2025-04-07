import os
from mlx_lm import load
from mlx_lm.generate import generate_step
import mlx.core as mx

# Carrega o modelo e o tokenizer
model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

# Diretório onde estão os arquivos .txt
directory_path = "txts"

# Lista todos os arquivos .txt no diretório
txt_files = [file for file in os.listdir(directory_path) if file.endswith(".txt")]

# Itera sobre cada arquivo .txt encontrado
for txt_file in txt_files:
    file_path = os.path.join(directory_path, txt_file)
    
    # Lê o conteúdo do arquivo
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    # Verifica se há ao menos uma linha no arquivo
    if len(lines) < 2:
        print(f"Arquivo {txt_file} não possui conteúdo suficiente (pergunta e resposta).")
        continue
    
    # Define a pergunta e a resposta
    pergunta = lines[0].strip()  # Primeira linha como pergunta
    resposta_contexto = "".join(lines[1:]).strip()  # Restante como contexto
    
    # Cria o prompt com o contexto e a pergunta
    prompt = resposta_contexto + "\n\n" + pergunta
    
    # Codifica o prompt e transforma em mlx.core.array
    input_tokens = mx.array(tokenizer.encode(prompt))
    
    # Gera os tokens de resposta
    output = ""
    for token, _ in generate_step(input_tokens, model, max_tokens=256):
        output += tokenizer.decode([token])
    
    print(f"\nArquivo: {txt_file}")
    print("Pergunta:", pergunta)
    print("Resposta gerada:\n", output)

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
