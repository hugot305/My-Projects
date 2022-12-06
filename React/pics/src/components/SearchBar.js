import './SearchBar.css';
import ImageUpload from './ImageUpload';
import { useState } from 'react';

function SearchBar({ onSubmit }) {
  const [term, setTerm] = useState('');

  const handleFormSubmit = (event) => {
    event.preventDefault();

    onSubmit(term);
  };

  const handleChange = (event) => {
    setTerm(event.target.value);
  };

  return (
    <div className="search-bar">
      <form onSubmit={handleFormSubmit}>
        <div className='row'>
          <div className='col-md-3'>
            <input value={term} placeholder="Search Images..." onChange={handleChange} />
          </div>
        </div>
      </form>
    </div>
  );
}

export default SearchBar;
