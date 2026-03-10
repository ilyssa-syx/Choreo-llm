import json
import os


def interleave(file_name, caption_dir, modifier_dir, interleaved_dir):
    with open(os.path.join(caption_dir, file_name), 'r') as f:
        caption_data = json.load(f)
    with open(os.path.join(modifier_dir, file_name), 'r') as f:
        modifier_data = json.load(f)
        
    
    interleaves = []
    
    
    for i in range(len(caption_data)):
        caption = caption_data[i]
        modifier = modifier_data[i]
        interleaved = []
        for j in range(max(len(caption['modifier']), len(modifier['modifier']))):
            if j < len(caption['modifier']):
                interleaved.append(caption['modifier'][j])
            if j < len(modifier['modifier']):
                interleaved.append(modifier['modifier'][j])
        interleave_item = caption
        interleave_item['modifier'] = interleaved
        interleaves.append(interleave_item)
    
    with open(os.path.join(interleaved_dir, file_name), 'w') as f:
        json.dump(interleaves, f, indent=4)


def process_dir():
    caption_dir = 'caption/train'
    modifier_dir = 'modifier/test'
    interleaved_dir = './interleaved/test'
    os.makedirs(interleaved_dir, exist_ok=True)

    for file_name in os.listdir(caption_dir):
        interleave(file_name, caption_dir, modifier_dir, interleaved_dir)

if __name__ == "__main__":
    process_dir()