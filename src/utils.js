function chunkText(text, chunkSize = 300) {
  // pecah teks menjadi array potongan per chunkSize kata
  const words = text.split(" ");
  const chunks = [];
  for (let i = 0; i < words.length; i += chunkSize) {
    chunks.push(words.slice(i, i + chunkSize).join(" "));
  }
  return chunks;
}

module.exports = { chunkText };
