"""Vectorized clean type conditioning and motif readout; identical parameters."""
import torch
from .fullgraph import FullGraphEncoder,collate_graphs,SCHEMA

def incidence(record):
    # Cache fixed membership only, never parameter-dependent embeddings/states.
    membership=tuple(tuple(m) for m in record['members'])
    if record.get('_incidence_members')!=membership:
        n=record['graph'].node_features.size(0)
        mat=torch.zeros(len(record['members']),n)
        for k,members in enumerate(record['members']):mat[k,members]=1
        record['_incidence_cpu']=mat
        record['_incidence_members']=membership
    return record['_incidence_cpu']

class FastFullGraphEncoder(FullGraphEncoder):
    def packed(self,items,device):
        n=max(r['graph'].node_features.size(0) for r in items)
        m=max(1,max(len(r['members']) for r in items))
        member=torch.zeros(len(items),m,n)
        types=torch.zeros(len(items),m,dtype=torch.long)
        valid=[]
        for b,r in enumerate(items):
            k=len(r['members']);a=r['graph'].node_features.size(0)
            member[b,:k,:a]=incidence(r)
            types[b,:k]=torch.tensor(r['types'],dtype=torch.long)
            valid.extend(b*m+j for j in range(k))
        return member.to(device),types.to(device),torch.tensor(valid,dtype=torch.long,device=device)

    def batch(self,items,device):
        batch=collate_graphs([r['graph'] for r in items]).to(device)
        member,types,_=self.packed(items,device)
        counts=member.sum(1)
        sums=torch.bmm(member.transpose(1,2),self.type_embedding(types))
        condition=torch.where(counts[...,None]>0,sums/counts.clamp_min(1)[...,None],self.type_embedding.weight[0])
        return batch,condition*batch.node_mask[...,None]

    def encode_records(self,items,batch_size=8,pair_budget=0):
        vectors=[]
        for start in range(0,len(items),batch_size):
            chunk=items[start:start+batch_size]
            n=max(r['graph'].node_features.size(0) for r in chunk)
            if pair_budget and len(chunk)*n*n>pair_budget:
                if len(chunk)==1:raise ValueError('Full molecule exceeds pair budget')
                vectors.append(self.encode_records(chunk,1,pair_budget));continue
            h=self(chunk)
            member,_,valid=self.packed(chunk,h.device)
            pooled=torch.bmm(member,h)/member.sum(2).clamp_min(1)[...,None]
            pooled=pooled.flatten(0,1).index_select(0,valid)
            if len(valid):vectors.append(self.pool_projection(pooled))
        return torch.cat(vectors) if vectors else self.type_embedding.weight.new_empty((0,self.config['hidden']))

def load_encoder(path,device='cpu'):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck['schema']!=SCHEMA:raise ValueError('Fullgraph schema mismatch')
    model=FastFullGraphEncoder(ck['model_config'],ck['vocab'])
    model.load_state_dict(ck['model'],strict=True)
    return model.to(device),ck
