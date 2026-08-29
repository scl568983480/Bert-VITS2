import torch
import utils
from infer import get_net_g, infer

config_path = "Data/voice_cn/config.json"
model_path = "Data/voice_cn/models/G_0.pth"
device = "cpu"

hps = utils.get_hparams_from_file(config_path)
version = getattr(hps, "version", "2.3")
print("version:", version, flush=True)
net_g = get_net_g(model_path, version, device, hps)
print("model loaded", flush=True)

sid = list(hps.data.spk2id.keys())[0]
print("sid:", sid, "id:", hps.data.spk2id[sid], flush=True)

text = "你好，这是一个测试。"
audio = infer(
    text=text,
    emotion=0,
    sdp_ratio=0.2,
    noise_scale=0.6,
    noise_scale_w=0.8,
    length_scale=1.0,
    sid=sid,
    language="ZH",
    hps=hps,
    net_g=net_g,
    device=device,
)
print("audio shape:", audio.shape, "sr:", hps.data.sampling_rate, flush=True)
try:
    import soundfile as sf
    sf.write("test_out.wav", audio, hps.data.sampling_rate)
    print("SAVED test_out.wav OK", flush=True)
except Exception as e:
    print("save failed:", repr(e), flush=True)
