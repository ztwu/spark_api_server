import org.pmml4s.model.Model;
import java.util.HashMap;
import java.util.Map;

public class PMML4SDemo {

    public static void main(String[] args) {

        Model model = Model.fromFile("lr.xml");

        Map<String, Object> result = model.predict(new HashMap<String, Object>() {{
            put("field_0", 2);
            put("field_1", 4);
            put("field_2", 1);
            put("field_3", 5);
        }});

        System.out.println(result);
        System.out.println(result.get("predicted_target"));

    }
}