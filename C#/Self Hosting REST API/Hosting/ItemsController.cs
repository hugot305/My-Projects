using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text;
using System.Web.Http;
namespace Hosting
{
     public class ItemsController : ApiController
    {
     Item[] items = new Item[]   
  
        {  
            new Item { Id = 1, Name = "Apple", Category = "Fruit" },  
            new Item{ Id = 2, Name = "Tomato", Category = "vasitable" },  
            new Item{ Id = 3, Name = "T-Shirt", Category = "cloths" }  
        };

     public IEnumerable<Item> GetAllItems()
        {
            return items;
        }

     public Item GetItemById(int id)
        {
            var item = items.FirstOrDefault((i) => i.Id == id);
            if (item == null)
            {
                throw new HttpResponseException(HttpStatusCode.NotFound);
            }
            return item;
        }

     public IEnumerable<Item> GetItemsByCategory(string category)
        {
            return items.Where(i => string.Equals(i.Category, category,
                    StringComparison.OrdinalIgnoreCase));
        }
    }
}
