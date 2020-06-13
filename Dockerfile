
FROM python:latest

# Download, extract, build, relocate and tidy up latte-int.
WORKDIR /root/
RUN wget https://github.com/latte-int/latte/releases/download/version_1_7_5/latte-integrale-1.7.5.tar.gz && \
    tar -xzf latte-integrale-1.7.5.tar.gz && \
    cd ./latte-integrale-1.7.5/ && \
    ./configure && \
    make && \
    cp -r dest/bin ../bin && \
    cd .. && \
    rm -rf latte-integrale-1.7.5*

# Install dev version of curver.
RUN pip install pip --upgrade
RUN pip install git+git://github.com/MarkCBell/curver.git@dev

# Copy scripts.
COPY *.py ./

# Run.
CMD python sample.py

# Build with:
# $ docker build -t markcbell/mirzakhani .
# Run with:
# $ docker run --rm -t markcbell/mirzakhani
