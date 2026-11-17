"""Constantes compartilhadas entre os módulos do EcoIA."""

SEMENTE = 42

# Classes de resíduo reconhecidas pelo modelo de Machine Learning.
CLASSES = ["plastico", "papel", "vidro", "metal", "outros"]

NOMES_EXIBICAO = {
    "plastico": "Plástico",
    "papel": "Papel/Papelão",
    "vidro": "Vidro",
    "metal": "Metal",
    "outros": "Outros/Rejeito",
}

# Cores de coleta seletiva conforme a Resolução CONAMA nº 275/2001.
LIXEIRAS = {
    "plastico": "vermelha",
    "papel": "azul",
    "vidro": "verde",
    "metal": "amarela",
    "outros": "cinza",
}

# Mapeamento das pastas do dataset TrashNet para as classes do projeto.
# cardboard (papelão) é agrupado em "papel" porque segue a mesma cor de lixeira;
# trash (rejeito) vira "outros".
MAPA_TRASHNET = {
    "plastic": "plastico",
    "paper": "papel",
    "cardboard": "papel",
    "glass": "vidro",
    "metal": "metal",
    "trash": "outros",
}

INTENCOES = [
    "onde_descartar",
    "cor_lixeira",
    "confirmar_descarte",
    "preparo",
    "reciclavel",
    "identificar_material",
    "fora_escopo",
]
