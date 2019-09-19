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
def a_deps(s, d):
    dep = Word(cond.get_word(s, d), cond.get_pos2(s, d))
    assert is_adjective(dep)
    h = cliqs.depgraph.head_of(s, d)
    head = Word(cond.get_word(s, h), cond.get_pos2(s, h))
    r = cliqs.depgraph.deptype_to_head_of(s, d)
    num_deps = len(list(cliqs.depgraph.dependents_of(s, d)))
    left = d < h
    return {
        'd_word': dep.word,
        'd_pos': dep.pos,
        'h_word': head.word,
        'h_pos': head.pos,
        'deptype': r,
        'num_deps': num_deps,
        'left': left,
    }
        
@assertion_filter
def aan_triples(s, n):
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
    return {'noun_word': head.word, 'noun_pos': head.pos, 'adj1_word': adj1.word, 'adj2_word': adj2.word, 'adj1_pos': adj1.pos, 'adj2_pos': adj2.pos}
                                
def extract(f, sentences):
    for s in sentences:
        for n in s.nodes():
            if n != 0:
                result = f(s, n)
                if result is not None:
                    yield result
                    
def main(mode='aan', limit=None):
    if limit is not None:
        limit = int(limit)
    import cliqs.readcorpora
    # This will read from stdin:
    corpus = cliqs.readcorpora.UniversalDependency1Treebank()
    if mode == 'aan':
        results = extract(aan_triples, corpus.sentences())
    elif mode == 'a':
        results = extract(a_deps, corpus.sentences())
    if limit is not None:
        results = rfutils.take(results, limit)
    rfutils.write_dicts(sys.stdout, results)
    return 0

if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
            
            
    
