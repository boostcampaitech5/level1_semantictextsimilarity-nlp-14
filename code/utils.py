""" 파일 설명
        - 다른 팀 코드를 뜯어보면서 공부한 파일입니다.
        - 해당 함수를 불러오는 형태로 필요한 부분을 뜯어서 고칠 수 있도록 def 형태로 선언해두었습니다.
        
    함수 설명
        - swap setence
            input_ids는 배치 사이즈로 들어온다는 것에 유의하기만 하면 됨.
            + nonzero를 통해서 SEP_ID토큰 위치 파악, full_like를 통해서 기존 배치 사이즈와 동일한 사이즈를 미리 만들어 놓는 것이 핵심.


"""

import torch
import transformers

SEP_ID = transformers.AutoTokenizer.from_pretrained("klue/roberta-small").sep_token_id
CLS_ID = transformers.AutoTokenizer.from_pretrained("klue/roberta-small").cls_token_id
PAD_ID = transformers.AutoTokenizer.from_pretrained("klue/roberta-small").pad_token_id

tokenizer = transformers.AutoTokenizer
tokenizer.add_special_tokens({"additional_special_tokens_ids":
    ["[petition]", "[nsmc]", "[slack]", "[sampled]", "[rtt]"]
    })
ALL_SPECIAL_IDS = set(
        [SEP_ID, CLS_ID] + tokenizer.additional_special_tokens_ids
    )



def swap_sentences(input_ids, attention_mask):  # 2 * (batch_size, max_seq_len)
    sep_poss = (input_ids == SEP_ID).nonzero(as_tuple=True)[1]
    sep_poss = [
        (sep_poss[i], sep_poss[i + 1]) for i in range(0, len(sep_poss) - 1, 2)  # start, end pos of sentence
    ]  # [(sep1, sep2),(sep1, sep2)]

    input_ids_swapped = torch.full_like(input_ids, fill_value=PAD_ID)   # 모든 부분을 PAD로 채운 뒤, 첫 열만 CLS로 변경
    input_ids_swapped[:, 0] = CLS_ID
    # token_type_ids_swapped = torch.zeros_like(token_type_ids)

    for i, sep_pos in enumerate(sep_poss):      # i = batch 에서의 number. sep2 = 아마 end of whole sentence
        sep_1, sep_2 = sep_pos
        input_ids_swapped[i, 1 : sep_2 - sep_1 + 1] = input_ids[
            i, sep_1 + 1 : sep_2 + 1
        ]
        # 두 번째 sentence(인코딩된) 를 앞에다가 배치
        input_ids_swapped[i, sep_2 - sep_1 + 1 : sep_2 + 1] = input_ids[
            i, 1 : sep_1 + 1
        ]
        # 첫 번째 sentence를 뒤에다가 배치
        # token_type_ids_swapped[i, sep_2 - sep_1 + 1 : sep_2 + 1] = 1

    return input_ids_swapped, attention_mask  # , token_type_ids_swapped



def weight_loss(loss_func, logits, ground_truth, penalty_zero, penalty_five, threshold=0.05):
    
    basic_loss = loss_func(reduction="none")(logits, ground_truth)
    
    # penalty for zero output
    penalty_mask_zero = (ground_truth != 0) & (logits <= threshold) # 아마 batch로 가져와서 True 를 반환할 것.
    penalty_loss = basic_loss * penalty_zero * penalty_mask_zero
    
    # panalty for five output
    penalty_mask_five = (ground_truth != 5) & (logits >= 5-threshold)
    penalty_loss += basic_loss * penalty_five * penalty_mask_five
    
    loss = basic_loss + penalty_loss
    return loss



def R_drop_L1(logits_1, logits_2, alpha):
    return torch.abs(logits_1 - logits_2).mean() * alpha


# 스페셜 토큰에 대한 hidden_state만을 가져온다.
def get_special_token_hidden_states(input_ids, hidden_states):
    batch_size = input_ids.size(0)
    hidden_size = hidden_states.size(-1)

    special_token_positions = torch.zeros_like(input_ids).bool()

    # bitwise 연산. 인코딩 된 input_id에서 special token id가 있는 위치는 True로 세팅된다.
    for token_id in ALL_SPECIAL_IDS:
        special_token_positions |= input_ids == token_id

    special_token_hidden_states = []
    for i in range(batch_size):
        batch_special_token_hidden_states = torch.masked_select(
            hidden_states[i], special_token_positions[i].unsqueeze(-1)
        )   # input, mask -> input에서 mask=True인 index가 1-d tensor로 반환된다.
        if args.pool_special_voting:
            batch_special_token_hidden_states = batch_special_token_hidden_states.view(
                -1, hidden_size
            )

        special_token_hidden_states.append(batch_special_token_hidden_states)

    # stack = Concatenates a sequence of tensors along a new dimension.
    # [tensor[], tensor[], tensor[] ... ] 이런 식이었다면, 이를 dim=0으로 다시 이어붙임. output 자체가 tensor 가 된다.
    # tensor[[], [], [], ...] -> (batch, special_token_number, hidden_state) 형태로 리턴
    return torch.stack(special_token_hidden_states, dim=0)
"""
참고:
special_tokens_dict = {
    "additional_special_tokens": [
        "[petition]",
        "[nsmc]",
        "[slack]",
        "[sampled]",
        "[rtt]",
    ]
}
self.tokenizer.add_special_tokens(
    special_tokens_dict
)  # Added new tokens to vocabulary.
        
        
ALL_SPECIAL_IDS = set(
        [SEP_ID, CLS_ID] + dataloader.tokenizer.additional_special_tokens_ids
    )
"""


elif args.pool_special_voting:
    special_hidden_states_1 = get_special_token_hidden_states(
        input_ids_1, hidden_states_1
    )  # (batches, num_special_tokens, hidden_size)

    special_token_logits_1 = [
        classifier(hidden).squeeze(-1)
        for hidden, classifier in zip(
            special_hidden_states_1.split(1, dim=1),
            self.pool_special_voting_token_heads,
        )
    ]

    # Stack the logits along dimension 1
    special_token_logits_1 = torch.stack(special_token_logits_1, dim=1)
    logits_1 = torch.sum(
        self.pool_special_voting_weight * special_token_logits_1, dim=1
    ).unsqueeze(1)

    del special_hidden_states_1
    del special_token_logits_1

    if args.S_swap:
        special_hidden_states_2 = get_special_token_hidden_states(
            input_ids_2, hidden_states_2
        )  # (batches, num_special_tokens, hidden_size)

        special_token_logits_2 = [
            classifier(hidden).squeeze(-1)
            for hidden, classifier in zip(
                special_hidden_states_2.split(1, dim=1),
                self.pool_special_voting_token_heads,
            )
        ]

        # Stack the logits along dimension 1
        special_token_logits_2 = torch.stack(special_token_logits_2, dim=1)
        logits_2 = torch.sum(
            self.pool_special_voting_weight * special_token_logits_2, dim=1
        ).unsqueeze(1)

        del special_hidden_states_2
        del special_token_logits_2