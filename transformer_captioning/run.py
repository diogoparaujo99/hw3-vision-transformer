
import argparse
import os

import torch
from utils import * 
from torch.utils.data import DataLoader
from trainer import Trainer
from transformer import SwiGLU, TransformerDecoder
from matplotlib import pyplot as plt

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
torch.use_deterministic_algorithms(True)
set_all_seeds(42) ### DO NOT CHANGE THIS LINE
configs = {
    'case1': (2, 2, 1e-4),
    'case2': (4, 6, 1e-4),
    'case3': (4, 6, 1e-3),
    'swiglu': (4, 6, 1e-4),
}
parser = argparse.ArgumentParser()
parser.add_argument('--config', choices=configs, default='case1')
exp_name = parser.parse_args().config
num_heads, num_layers, learning_rate = configs[exp_name]

train_dataset = CocoDataset(load_coco_data(max_train=1024), 'train')
train_dataloader =  DataLoader(train_dataset, batch_size=64)

val_dataset = CocoDataset(load_coco_data(max_val = 1024), 'val')
val_dataloader =  DataLoader(val_dataset, batch_size=64)


device = 'cuda'
transformer = TransformerDecoder(
          word_to_idx=train_dataset.data['word_to_idx'],
          idx_to_word = train_dataset.data['idx_to_word'],
          input_dim=train_dataset.data['train_features'].shape[1],
          embed_dim=256,
          num_heads=num_heads,
          num_layers=num_layers,
          max_length=30,
          device = device
        )

if exp_name == 'swiglu':
    for layer in transformer.layers:
        mlp = layer.feedforward_block.mlp
        embed_dim = mlp[0].in_features
        hidden_dim = 2 * mlp[0].out_features // 3
        layer.feedforward_block.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 2 * hidden_dim),
            SwiGLU(),
            mlp[2],
            torch.nn.Linear(hidden_dim, embed_dim),
        ).apply(transformer._init_weights).to(device)

trainer = Trainer(transformer, train_dataloader, val_dataloader,
          num_epochs=100,
          learning_rate=learning_rate,
          device = device
        )

trainer.train()
print(f'Final training loss: {trainer.loss_history[-1]:.6f}')
minimum_loss = min(trainer.loss_history)
min_epoch = trainer.loss_history.index(minimum_loss) + 1
print(f'Minimum training loss: {minimum_loss:.6f} (epoch {min_epoch})')

# Plot the training losses.
epochs = range(1, len(trainer.loss_history) + 1)
plt.plot(epochs, trainer.loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
os.makedirs('plots', exist_ok=True)
plt.title('Training loss history')
if exp_name == 'case3' and len(epochs) >= 40:
    zoom = plt.gca().inset_axes([0.40, 0.47, 0.55, 0.43])
    zoom.plot(epochs[39:], trainer.loss_history[39:])
    zoom.set_title(f'Epochs 40–{epochs[-1]}', fontsize=10)
    zoom.set_xticks([40, (40 + epochs[-1]) // 2, epochs[-1]])
    zoom.tick_params(labelsize=9)
plt.savefig('plots/' + exp_name + '_loss_out.png')
plt.close()


def vis_imgs(split):
    data = {'train': train_dataset.data, 'val': val_dataset.data}[split]
    loader = {'train': train_dataloader, 'val': val_dataloader}[split]
    num_imgs = 0 
    for batch in loader:
      features, gt_captions, idxs = batch
      urls = data["%s_urls" % split][idxs]
      
      gt_captions = decode_captions(gt_captions, transformer.idx_to_word)
      sample_captions = transformer.sample(features, max_length=30)
      sample_captions = decode_captions(sample_captions, transformer.idx_to_word)
      
      for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
          img = image_from_url(url)
          # Skip missing URLs.
          if img is not None: 
            plt.figure()
            plt.imshow(img)            
            plt.title('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))
            plt.axis('off')
            plt.savefig('plots/' + exp_name + '_%s_%d.png' % (split, num_imgs),
                        bbox_inches='tight', dpi=140)
            plt.close()
            num_imgs += 1
            if num_imgs >= 5:
                return

transformer.eval()
vis_imgs('train')
vis_imgs('val')
