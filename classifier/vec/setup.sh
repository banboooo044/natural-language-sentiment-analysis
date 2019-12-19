#!/bin/sh
spin(){
    spinner="/|\\-/|\\-"
    while :
    do
    for i in `seq 0 7`
        do
            echo -n "${spinner:$i:1}"
            echo -en "\010"
            sleep 0.5
    done
    done
}

echo "setup ..."

echo "create_label.py"
spin &
SPIN_PID=$!
trap "kill -9 $SPIN_PID" `seq 0 15`
python3 ./create_label.py
kill -9 $SPIN_PID

echo "bow.py"
spin &
SPIN_PID=$!
trap "kill -9 $SPIN_PID" `seq 0 15`
python3 ./bow.py
kill -9 $SPIN_PID

echo "bow_tf-idf.py"
spin &
SPIN_PID=$!
trap "kill -9 $SPIN_PID" `seq 0 15`
python3 ./bow_tf-idf.py
kill -9 $SPIN_PID

echo "get_embedding_matrix.py"
spin &
SPIN_PID=$!
trap "kill -9 $SPIN_PID" `seq 0 15`
python3 ./get_embedding_matrix.py word2vec
python3 ./get_embedding_matrix.py fasttext
kill -9 $SPIN_PID

echo "get_doc2vec_matrix.py"
spin &
SPIN_PID=$!
trap "kill -9 $SPIN_PID" `seq 0 15`
python3 ./get_doc2vec_matrix.py dbow
python3 ./get_doc2vec_matrix.py dmpv
kill -9 $SPIN_PID

echo "get_scdv.py"
spin &
SPIN_PID=$!
trap "kill -9 $SPIN_PID" `seq 0 15`
python3 ./get_scdv.py
kill -9 $SPIN_PID

echo "get_sdv.py"
spin &
SPIN_PID=$!
trap "kill -9 $SPIN_PID" `seq 0 15`
python3 ./sdv.py
kill -9 $SPIN_PID

echo "OK"