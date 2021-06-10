using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Net.Http;
namespace Client
{
    class Program
    {
        static HttpClient client = new HttpClient();
        static void Main(string[] args)
        {
            client.BaseAddress = new Uri("http://localhost:8080");

            ListAllItems();
            ListItem(1);
            ListItems("fruit");

            Console.WriteLine("Press Enter to quit.");
            Console.ReadLine();

        }
        static void ListAllItems()
        {
            HttpResponseMessage resp = client.GetAsync("api/items").Result;
            //resp.EnsureSuccessStatusCode();

            var items = resp.Content.ReadAsAsync<IEnumerable<Hosting.Item>>().Result;
            foreach (var i in items)
            {
                Console.WriteLine("{0} {1} {2}", i.Id, i.Name,  i.Category);
            }
        }

        static void ListItem(int id)
        {
            var resp = client.GetAsync(string.Format("api/products/{0}", id)).Result;
            //resp.EnsureSuccessStatusCode();

            var item = resp.Content.ReadAsAsync<Hosting.Item>().Result;
            Console.WriteLine("ID {0}: {1}", id, item.Name);
        }

        static void ListItems(string category)
        {
            Console.WriteLine("items in '{0}':", category);

            string query = string.Format("api/items?category={0}", category);

            var resp = client.GetAsync(query).Result;
            resp.EnsureSuccessStatusCode();

            var items = resp.Content.ReadAsAsync<IEnumerable<Hosting.Item>>().Result;
            foreach (var item in items)
            {
                Console.WriteLine(item.Name);
            }
        }
    }
}
