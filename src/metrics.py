import torch
import re

def extract_spans_from_bio(seq, id2slot):
    """Extracts (slot_type, start_idx, end_idx) from a sequence of BIO tags."""
    spans = []
    current_span = None
    for i, tag_id in enumerate(seq):
        tag = id2slot.get(tag_id, 'O')
        if tag == 'O' or tag == '<PAD>':
            if current_span:
                spans.append(current_span)
                current_span = None
        elif tag.startswith('B-'):
            if current_span:
                spans.append(current_span)
            current_span = (tag[2:], i, i)
        elif tag.startswith('I-'):
            if current_span and current_span[0] == tag[2:] and current_span[2] == i - 1:
                current_span = (current_span[0], current_span[1], i)
            else:
                if current_span:
                    spans.append(current_span)
                current_span = (tag[2:], i, i)
    if current_span:
        spans.append(current_span)
    return spans

def extract_spans_from_annot_utt(annot_utt):
    """Extracts ground truth (slot_type, text_span) from raw annot_utt."""
    spans = []
    pattern = r'\[(.*?):\s*(.*?)\]'
    matches = re.findall(pattern, annot_utt)
    for slot_type, text_span in matches:
        spans.append((slot_type.strip(), text_span.strip()))
    return spans

def get_predicted_text_spans(bio_spans, raw_utt_words):
    """Converts index spans to text strings using the raw utterance."""
    text_spans = []
    for slot_type, start, end in bio_spans:
        if start < len(raw_utt_words) and end < len(raw_utt_words):
            span_text = " ".join(raw_utt_words[start:end+1])
            text_spans.append((slot_type, span_text))
    return text_spans

def evaluate_model(model, dataloader, device, id2slot):
    """Evaluates the model and returns formatted percentage metrics."""
    model.eval()
    
    intent_correct = 0
    intent_total = 0
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            seq_lens = batch['seq_lens'].to(device)
            intent_labels = batch['intent_label'].to(device)
            
            _, intent_logits, slot_logits = model(
                input_ids, 
                attention_mask=attention_mask, 
                seq_lens=seq_lens
            )
            
            # 1. Intent Accuracy Calculation
            intent_preds = torch.argmax(intent_logits, dim=-1)
            intent_correct += (intent_preds == intent_labels).sum().item()
            intent_total += intent_labels.size(0)
            
            # 2. Exact-Match Slot Micro-F1 Calculation
            slot_preds = torch.argmax(slot_logits, dim=-1).cpu().numpy()
            raw_utts = batch['raw_utt']
            raw_annot_utts = batch['raw_annot_utt']
            
            for i in range(len(raw_utts)):
                seq_len = seq_lens[i].item()
                pred_seq = slot_preds[i][:seq_len]
                pred_bio_spans = extract_spans_from_bio(pred_seq, id2slot)
                
                raw_words = raw_utts[i].split()
                pred_text_spans = get_predicted_text_spans(pred_bio_spans, raw_words)
                pred_set = set(pred_text_spans)
                
                true_text_spans = extract_spans_from_annot_utt(raw_annot_utts[i])
                true_set = set(true_text_spans)
                
                total_tp += len(pred_set.intersection(true_set))
                total_fp += len(pred_set - true_set)
                total_fn += len(true_set - pred_set)

    # 3. Final Metric Formatting
    intent_accuracy = (intent_correct / intent_total) * 100 if intent_total > 0 else 0.0
    
    denominator = (2 * total_tp + total_fp + total_fn)
    slot_micro_f1 = (100 * (2 * total_tp) / denominator) if denominator > 0 else 0.0
    
    combined_score = (intent_accuracy + slot_micro_f1) / 2
    
    return {
        'intent_accuracy': round(intent_accuracy, 2),
        'slot_micro_f1': round(slot_micro_f1, 2),
        'combined_score': round(combined_score, 2)
    }