# ⚡ qVicuna Flash

Ultra hızlı AI chat sistemi - Replicate üzerinde çalışan optimize edilmiş Vicuna-7B modeli.

## 🚀 Özellikler

- **Flash Mode**: Ultra hızlı yanıtlar
- **Standard Mode**: Daha kaliteli yanıtlar  
- **GPU Optimized**: RTX 4090 ile optimize edildi
- **Easy API**: Basit REST API

## 🧪 Kullanım

```python
import replicate

output = replicate.run(
    "semihpolat/qvicuna-flash",
    input={
        "message": "Merhaba!",
        "flash_mode": True,
        "max_tokens": 50
    }
)

print(output)
```

## 🔥 Deploy

GitHub Actions ile otomatik deploy edilir.

## 📊 Performance

- **Flash Mode**: ~100-200ms
- **Standard Mode**: ~300-500ms
- **Model**: Vicuna-7B optimized
