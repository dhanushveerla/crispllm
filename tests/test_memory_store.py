"""Unit tests for EdgeMem Embedded SQLite Tiered Memory Store and Extractor."""

import concurrent .futures 
from pathlib import Path 
import pytest 

from edgemem .memory .extractor import ExtractedFact ,extract_facts_from_messages 
from edgemem .memory .store import Fact ,MemoryStore 


def test_wal_and_in_memory_modes (tmp_path :Path )->None :
    """Test WAL creation on a temp file DB and :memory: operation."""
    disk_db =tmp_path /"sub"/"memory.db"
    with MemoryStore (disk_db )as store :
        assert store .journal_mode =="wal"
        store .upsert_fact ("sess_wal","KEY_A","VAL_A",kind ="note")
        facts =store .list_facts ("sess_wal")
        assert len (facts )==1 
        assert facts [0 ].key =="KEY_A"


    with MemoryStore (":memory:")as mem_store :
        assert mem_store .journal_mode =="memory"
        mem_store .upsert_fact ("sess_mem","KEY_M","VAL_M",kind ="note")
        facts_mem =mem_store .list_facts ("sess_mem")
        assert len (facts_mem )==1 
        assert facts_mem [0 ].key =="KEY_M"


def test_upsert_overwrites_without_duplicates_or_stale_fts ()->None :
    """Test upsert_fact overwrites on (session_id, kind, key) without duplicate rows or stale FTS."""
    with MemoryStore (":memory:")as store :
        fid1 =store .upsert_fact (
        "sess1","MAX_WORKERS","4",kind ="constant",scope ="session",pinned =False 
        )
        assert store .stats ()["facts_count"]==1 


        res_old =store .search_facts ("4",session_id ="sess1")
        assert len (res_old )==1 
        assert res_old [0 ].value =="4"


        fid2 =store .upsert_fact (
        "sess1","MAX_WORKERS","16",kind ="constant",scope ="session",pinned =True 
        )
        assert fid2 !=fid1 
        assert store .stats ()["facts_count"]==1 


        res_stale =store .search_facts ("4",session_id ="sess1")
        assert len (res_stale )==0 


        res_new =store .search_facts ("16",session_id ="sess1")
        assert len (res_new )==1 
        assert res_new [0 ].value =="16"
        assert res_new [0 ].pinned is True 


        res_key =store .search_facts ("MAX_WORKERS",session_id ="sess1")
        assert len (res_key )==1 
        assert res_key [0 ].value =="16"


def test_fts_search_bm25_and_hostile_punctuation ()->None :
    """Test FTS5 search handles BM25 ranking and hostile punctuation safely."""
    with MemoryStore (":memory:")as store :
        store .upsert_fact ("s1","K_LOW","apple fruit common food",kind ="note")
        store .upsert_fact ("s1","K_HIGH","apple apple apple fruit apple food delicious",kind ="note")
        store .upsert_fact ("s1","K_OTHER","banana cherry date",kind ="note")
        store .upsert_fact ("s2","K_ISO","apple in other session",kind ="note")


        results =store .search_facts ("apple",session_id ="s1",include_global =False )
        assert len (results )==2 
        assert results [0 ].key =="K_HIGH"
        assert results [1 ].key =="K_LOW"


        hostile_queries =[
        'foo::bar (baz-1) "unclosed [error]',
        '::: - ()',
        'SELECT * FROM facts WHERE id = 1;',
        '!@#$%^&*()_+=~`{}[]|\\:;"<>,.?/',
        '',
        '   ',
        'apple OR (NOT banana)',
        ]
        for hq in hostile_queries :

            matches =store .search_facts (hq ,session_id ="s1")
            assert isinstance (matches ,list )


def test_crud_episodes_clear_and_stats ()->None :
    """Test delete_fact, add_episode, clear_session, clear_all, and stats."""
    with MemoryStore (":memory:")as store :
        f1 =store .upsert_fact ("s1","C1","Must be frozen",kind ="constraint",pinned =True )
        f2 =store .upsert_fact ("s1","C2","Must be async",kind ="constraint",pinned =True )
        f3 =store .upsert_fact ("s2","C3","Must be singleton",kind ="constraint",pinned =False )
        f_global =store .upsert_fact (
        "s1","G1","Global invariant",kind ="constraint",scope ="global",pinned =True 
        )

        e1 =store .add_episode ("s1","user","User requested schema change",raw_tokens =120 )
        e2 =store .add_episode ("s2","assistant","Created initial plan",raw_tokens =85 )
        assert e1 >0 and e2 >0 


        st =store .stats ()
        assert st ["facts_count"]==4 
        assert st ["episodes_count"]==2 
        assert st ["sessions_count"]==2 


        pinned_s1 =store .get_pinned_facts ("s1",include_global =True )
        keys_s1 ={f .key for f in pinned_s1 }
        assert keys_s1 =={"C1","C2","G1"}

        pinned_s2 =store .get_pinned_facts ("s2",include_global =True )
        keys_s2 ={f .key for f in pinned_s2 }
        assert keys_s2 =={"G1"}


        assert store .delete_fact (f2 )is True 
        assert store .delete_fact (f2 )is False 
        assert len (store .list_facts ("s1"))==2 


        del_count =store .clear_session ("s1")
        assert del_count ==2 
        assert len (store .list_facts ("s1"))==0 
        assert len (store .list_episodes ("s1"))==0 
        assert len (store .list_facts ("s2"))==1 
        assert len (store .list_episodes ("s2"))==1 


        store .clear_all ()
        assert store .stats ()["facts_count"]==0 
        assert store .stats ()["episodes_count"]==0 


def test_extractor_with_multi_turn_and_distractor_turns ()->None :
    """Test extractor on 5-turn conversation: extracts constraints and code facts, ignores distractors."""
    messages =[

    {
    "role":"user",
    "content":(
    "Requirements for the service:\n"
    "- The TransactionRecord must be frozen and immutable.\n"
    "- Invariant: timestamps must be strictly monotonic.\n"
    "- The DatabaseManager must be a singleton.\n"
    "- Never delete audit records without explicit archiving."
    ),
    },

    {
    "role":"assistant",
    "content":(
    "Understood. Here is the scaffolding for the types and constants:\n"
    "```python\n"
    "MAX_RETRIES = 5\n"
    "DEFAULT_TIMEOUT = 30\n"
    "\n"
    "def calculate_hash(data: bytes, salt: str = '') -> str:\n"
    "    return 'hash_val'\n"
    "\n"
    "class TransactionRecord:\n"
    "    def __init__(self, tx_id: str, amount: int):\n"
    "        self.tx_id = tx_id\n"
    "        self.amount = amount\n"
    "```"
    ),
    },

    {
    "role":"user",
    "content":"Can you tell me how many planets are in the solar system? Also what is a good recipe for pizza?",
    },

    {
    "role":"assistant",
    "content":"There are 8 planets in our solar system. For pizza, mix flour, yeast, water, and let the dough rise for 24 hours.",
    },

    {
    "role":"user",
    "content":"Now implement the commit_transaction function following the previous constraints.",
    },
    ]

    extracted =extract_facts_from_messages (messages )
    assert len (extracted )>0 


    constraints =[f for f in extracted if f .kind =="constraint"]
    constants =[f for f in extracted if f .kind =="constant"]
    signatures =[f for f in extracted if f .kind =="signature"]


    assert len (constraints )>=3 
    for c in constraints :
        assert c .pinned is True 
        assert c .source_turn ==0 

    constraint_texts =" ".join (c .value for c in constraints )
    assert "frozen and immutable"in constraint_texts 
    assert "strictly monotonic"in constraint_texts 
    assert "singleton"in constraint_texts 
    assert "Never delete audit records"in constraint_texts 


    const_keys ={c .key for c in constants }
    assert "MAX_RETRIES"in const_keys 
    assert "DEFAULT_TIMEOUT"in const_keys 

    sig_keys ={s .key for s in signatures }
    assert "calculate_hash"in sig_keys 
    assert "TransactionRecord"in sig_keys 


    distractor_turns ={f .source_turn for f in extracted }
    assert 2 not in distractor_turns 
    assert 3 not in distractor_turns 
    assert 4 not in distractor_turns 


    with MemoryStore (":memory:")as store :
        for f in extracted :
            store .upsert_fact (
            session_id ="test_sess",
            key =f .key ,
            value =f .value ,
            kind =f .kind ,
            scope =f .scope ,
            source_turn =f .source_turn ,
            pinned =f .pinned ,
            )

        pinned =store .get_pinned_facts ("test_sess")
        assert len (pinned )==len (constraints )

        search_res =store .search_facts ("MAX_RETRIES",session_id ="test_sess")
        assert len (search_res )==1 
        assert "5"in search_res [0 ].value 


def test_extractor_regex_fallback_and_caps ()->None :
    """Test extractor handles un-fenced or partial code snippets and enforces caps."""
    messages =[
    {
    "role":"user",
    "content":"Rule: All network calls must be strictly logged. Configuration:\nBUFFER_LIMIT = 1024\ndef flush_buffer(force: bool = False):\n    pass",
    },
    {
    "role":"user",
    "content":"Final task prompt.",
    },
    ]
    extracted =extract_facts_from_messages (messages ,max_constraints =1 ,max_code_facts =1 )
    constraints =[f for f in extracted if f .kind =="constraint"]
    code_facts =[f for f in extracted if f .kind !="constraint"]

    assert len (constraints )==1 
    assert len (code_facts )==1 


def test_thread_concurrency_safety ()->None :
    """Test concurrent multi-threaded writes to MemoryStore do not raise errors."""
    with MemoryStore (":memory:")as store :
        def worker (w_id :int )->None :
            for i in range (20 ):
                store .upsert_fact (
                session_id =f"thread_{w_id }",
                key =f"KEY_{i }",
                value =f"VAL_{w_id }_{i }",
                kind ="note",
                )
                store .add_episode (
                session_id =f"thread_{w_id }",
                role ="user",
                summary =f"Worker {w_id } turn {i }",
                )

        with concurrent .futures .ThreadPoolExecutor (max_workers =8 )as executor :
            futures =[executor .submit (worker ,wid )for wid in range (8 )]
            for fut in futures :
                fut .result ()

        st =store .stats ()
        assert st ["facts_count"]==8 *20 
        assert st ["episodes_count"]==8 *20 
        assert st ["sessions_count"]==8 
