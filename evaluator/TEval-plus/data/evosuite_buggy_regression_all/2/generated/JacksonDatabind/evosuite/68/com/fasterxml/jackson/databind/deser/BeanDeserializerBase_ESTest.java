/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:03:02 GMT 2023
 */

package com.fasterxml.jackson.databind.deser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.TreeNode;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.deser.AbstractDeserializer;
import com.fasterxml.jackson.databind.ext.CoreXMLDeserializers;
import com.fasterxml.jackson.databind.jsontype.impl.AsArrayTypeDeserializer;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.BooleanNode;
import com.fasterxml.jackson.databind.node.DecimalNode;
import com.fasterxml.jackson.databind.node.FloatNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.LongNode;
import com.fasterxml.jackson.databind.node.NumericNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.node.ShortNode;
import com.fasterxml.jackson.databind.node.TextNode;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.Stack;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BeanDeserializerBase_ESTest extends BeanDeserializerBase_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(true);
      Stack<JsonNode> stack0 = new Stack<JsonNode>();
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0, stack0);
      Class<RuntimeException> class0 = RuntimeException.class;
      try { 
        objectMapper0.treeToValue((TreeNode) arrayNode0, class0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not deserialize instance of java.lang.RuntimeException out of START_ARRAY token
         //  at [Source: java.lang.String@0000000106; line: -1, column: -1]
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      BooleanNode booleanNode0 = BooleanNode.getFalse();
      Class<ObjectIdGenerators.StringIdGenerator> class0 = ObjectIdGenerators.StringIdGenerator.class;
      MapperFeature[] mapperFeatureArray0 = new MapperFeature[5];
      MapperFeature mapperFeature0 = MapperFeature.USE_STD_BEAN_NAMING;
      mapperFeatureArray0[0] = mapperFeature0;
      mapperFeatureArray0[1] = mapperFeatureArray0[0];
      mapperFeatureArray0[2] = mapperFeature0;
      mapperFeatureArray0[3] = mapperFeatureArray0[1];
      MapperFeature mapperFeature1 = MapperFeature.DEFAULT_VIEW_INCLUSION;
      mapperFeatureArray0[4] = mapperFeature1;
      ObjectMapper objectMapper1 = objectMapper0.disable(mapperFeatureArray0);
      try { 
        objectMapper1.treeToValue((TreeNode) booleanNode0, class0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not construct instance of com.fasterxml.jackson.annotation.ObjectIdGenerators$StringIdGenerator: no boolean/Boolean-argument constructor/factory method to deserialize from boolean value (false)
         //  at [Source: java.lang.String@0000000106; line: -1, column: -1]
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      objectMapper0.enableDefaultTyping();
      Class<AbstractDeserializer> class0 = AbstractDeserializer.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      DeserializationFeature deserializationFeature0 = DeserializationFeature.FAIL_ON_READING_DUP_TREE_KEY;
      ObjectReader objectReader0 = objectMapper0.reader(deserializationFeature0);
      LongNode longNode0 = new LongNode((-9223372036854775808L));
      JsonParser jsonParser0 = objectReader0.treeAsTokens(longNode0);
      Class<IOException> class0 = IOException.class;
      try { 
        objectMapper0.readValue(jsonParser0, class0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not construct instance of java.io.IOException: no long/Long-argument constructor/factory method to deserialize from Number value (-9223372036854775808)
         //  at [Source: java.lang.String@0000000106; line: -1, column: -1]
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0, 2043);
      ObjectNode objectNode0 = arrayNode0.objectNode();
      ObjectNode objectNode1 = objectNode0.put("fviw99<_3n)P", (-669));
      Class<ObjectIdGenerators.StringIdGenerator> class0 = ObjectIdGenerators.StringIdGenerator.class;
      try { 
        objectMapper0.treeToValue((TreeNode) objectNode1, class0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Unrecognized field \"fviw99<_3n)P\" (class com.fasterxml.jackson.annotation.ObjectIdGenerators$StringIdGenerator), not marked as ignorable (0 known properties: ])
         //  at [Source: java.lang.String@0000000106; line: -1, column: -1] (through reference chain: com.fasterxml.jackson.annotation.ObjectIdGenerators$StringIdGenerator[\"fviw99<_3n)P\"])
         //
         verifyException("com.fasterxml.jackson.databind.exc.UnrecognizedPropertyException", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<ObjectIdGenerators.IntSequenceGenerator> class0 = ObjectIdGenerators.IntSequenceGenerator.class;
      CoreXMLDeserializers.Std coreXMLDeserializers_Std0 = new CoreXMLDeserializers.Std(class0, 914);
      Class<AsArrayTypeDeserializer> class1 = AsArrayTypeDeserializer.class;
      try { 
        objectMapper0.convertValue((Object) coreXMLDeserializers_Std0, class1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not construct instance of com.fasterxml.jackson.databind.jsontype.impl.AsArrayTypeDeserializer: no suitable constructor found, can not deserialize from Object value (missing default constructor or creator, or perhaps need to add/enable type information?)
         //  at [Source: java.lang.String@0000001493; line: -1, column: -1]
         //
         verifyException("com.fasterxml.jackson.databind.ObjectMapper", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      ShortNode shortNode0 = new ShortNode((short)270);
      Class<ObjectIdGenerators.StringIdGenerator> class0 = ObjectIdGenerators.StringIdGenerator.class;
      try { 
        objectMapper0.treeToValue((TreeNode) shortNode0, class0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not construct instance of com.fasterxml.jackson.annotation.ObjectIdGenerators$StringIdGenerator: no int/Int-argument constructor/factory method to deserialize from Number value (270)
         //  at [Source: java.lang.String@0000000106; line: -1, column: -1]
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      BigInteger bigInteger0 = BigInteger.ZERO;
      Class<AsArrayTypeDeserializer> class0 = AsArrayTypeDeserializer.class;
      try { 
        objectMapper0.convertValue((Object) bigInteger0, class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not construct instance of com.fasterxml.jackson.databind.jsontype.impl.AsArrayTypeDeserializer: no suitable creator method found to deserialize from Number value (0)
         //  at [Source: java.lang.String@0000001069; line: -1, column: -1]
         //
         verifyException("com.fasterxml.jackson.databind.ObjectMapper", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      TextNode textNode0 = arrayNode0.textNode("]C");
      Class<AsArrayTypeDeserializer> class0 = AsArrayTypeDeserializer.class;
      try { 
        objectMapper0.convertValue((Object) textNode0, class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not construct instance of com.fasterxml.jackson.databind.jsontype.impl.AsArrayTypeDeserializer: no String-argument constructor/factory method to deserialize from String value (']C')
         //  at [Source: java.lang.String@0000000231; line: -1, column: -1]
         //
         verifyException("com.fasterxml.jackson.databind.ObjectMapper", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      Class<ObjectIdGenerators.StringIdGenerator> class0 = ObjectIdGenerators.StringIdGenerator.class;
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      ArrayNode arrayNode0 = objectNode0.putArray((String) null);
      NumericNode numericNode0 = arrayNode0.numberNode(891.782);
      try { 
        objectMapper0.treeToValue((TreeNode) numericNode0, class0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not construct instance of com.fasterxml.jackson.annotation.ObjectIdGenerators$StringIdGenerator: no double/Double-argument constructor/factory method to deserialize from Number value (891.782)
         //  at [Source: java.lang.String@0000000106; line: -1, column: -1]
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      BigDecimal bigDecimal0 = new BigDecimal((-897.3093));
      DecimalNode decimalNode0 = new DecimalNode(bigDecimal0);
      Class<CoreXMLDeserializers.Std> class0 = CoreXMLDeserializers.Std.class;
      try { 
        objectMapper0.treeToValue((TreeNode) decimalNode0, class0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not construct instance of com.fasterxml.jackson.databind.ext.CoreXMLDeserializers$Std: no suitable creator method found to deserialize from Number value (-897.3093000000000074578565545380115509033203125)
         //  at [Source: java.lang.String@0000000106; line: -1, column: -1]
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      FloatNode floatNode0 = FloatNode.valueOf(122.0F);
      Class<ObjectIdGenerators.StringIdGenerator> class0 = ObjectIdGenerators.StringIdGenerator.class;
      try { 
        objectMapper0.treeToValue((TreeNode) floatNode0, class0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not construct instance of com.fasterxml.jackson.annotation.ObjectIdGenerators$StringIdGenerator: no double/Double-argument constructor/factory method to deserialize from Number value (122.0)
         //  at [Source: java.lang.String@0000000106; line: -1, column: -1]
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      BooleanNode booleanNode0 = BooleanNode.getTrue();
      Class<ObjectIdGenerators.StringIdGenerator> class0 = ObjectIdGenerators.StringIdGenerator.class;
      try { 
        objectMapper0.treeToValue((TreeNode) booleanNode0, class0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not construct instance of com.fasterxml.jackson.annotation.ObjectIdGenerators$StringIdGenerator: no boolean/Boolean-argument constructor/factory method to deserialize from boolean value (true)
         //  at [Source: java.lang.String@0000000106; line: -1, column: -1]
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }
}