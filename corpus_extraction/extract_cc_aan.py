import sys

import rfutils
import cliqs.depgraph
import cliqs.conditioning as cond

from process_an_arcs import Word, is_adjective, is_noun, rep

def assertion_filter(f):
    def wrapper(*a, **k):
        try:
            return f(*a, **k)
        except AssertionError:
            return None
    return wrapper
        
@assertion_filter
def extract_triples_under_node(s, n):
    # It has to have at least 2 dependents to the left:
    deps = sorted(cliqs.depgraph.left_dependents_of(s, n))
    assert len(deps) > 1
    penult, last = deps[-2], deps[-1]
    
    # The head has to be a noun:
    head = Word(cond.get_word(s, n), cond.get_pos2(s, n))
    assert is_noun(head)

    # Check adjacency:
    assert last == n - 1 # immediately adjacent to noun
    assert penult == n - 2 # immediately adjacent to other adj

    # Check no dependents:
    assert not list(cliqs.depgraph.dependents_of(s, last)) 
    assert not list(cliqs.depgraph.dependents_of(s, penult))
    
    # Check adjectivity:
    adj2 = Word(cond.get_word(s, last), cond.get_pos2(s, last))
    assert is_adjective(adj2)

    adj1 = Word(cond.get_word(s, penult), cond.get_pos2(s, penult))
    assert is_adjective(adj1)

    # If all asserts have passed, we have a legitimate triple
    return head, adj1, adj2
                                
def extract_triples_from_sentence(s):
    for n in s.nodes():
        if n != 0:
            yield extract_triples_under_node(s, n)

def extract_triples(sentences):
    return filter(None, rfutils.flatmap(extract_triples_from_sentence, sentences))

def main(limit=None):
    if limit is not None:
        limit = int(limit)
    import cliqs.readcorpora
    # This will read from stdin:
    corpus = cliqs.readcorpora.UniversalDependency1Treebank()
    triples = extract_triples(corpus.sentences())
    if limit is not None:
        triples = rfutils.take(triples, limit)
    for n, a1, a2 in triples:
        print("\t".join(map(rep, [n, a1, a2])))
    return 0

if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
            
            
    
