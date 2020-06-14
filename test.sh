echo "Testing fast"
timeout 1m docker run --rm -t markcbell/mirzakhani-fast
echo "Testing classic"
timeout 1m docker run --rm -t markcbell/mirzakhani
exit 0
