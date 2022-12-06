import { useState } from 'react';
import SearchBar from './components/SearchBar';
import ImageList from './components/ImageList';
import ImageUpload from './components/ImageUpload';
import searchImages from './api';

import ImageCounter from './components/ImageCounter';
import 'bootstrap/dist/css/bootstrap.min.css';

function App() {
  const [images, setImages] = useState([]);

  const handleSubmit = async (term) => {
    const result = await searchImages(term);
    setImages(result);
  };

  let styles = {
    marginTop: '10px',
  };

  return (
    <div>
      <div className="row">
        <div className="col-2">
          <SearchBar onSubmit={handleSubmit} />
        </div>
        <div className='col-6'>
            <div style={styles}>
              <ImageUpload/>
            </div>

        </div>
      </div>
      <ImageCounter images={images}/>
      <ImageList images={images} />
    </div>
  );
}

export default App;
