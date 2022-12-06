
function ImageCounter({ images }) {
  if (images.length > 0)
    return <h2>{images.length} images</h2>
}

export default ImageCounter;
