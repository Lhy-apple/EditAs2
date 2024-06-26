/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:03:10 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.TreeNode;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.std.NumberDeserializers;
import com.fasterxml.jackson.databind.jsontype.TypeDeserializer;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.FloatNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class NumberDeserializers_ESTest extends NumberDeserializers_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<Long> class0 = Long.class;
      Long long0 = new Long(1271L);
      NumberDeserializers.LongDeserializer numberDeserializers_LongDeserializer0 = new NumberDeserializers.LongDeserializer(class0, long0);
      assertTrue(numberDeserializers_LongDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<Long> class0 = Long.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<Integer> class0 = Integer.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      NumberDeserializers.IntegerDeserializer numberDeserializers_IntegerDeserializer0 = new NumberDeserializers.IntegerDeserializer(class0, (Integer) null);
      assertTrue(numberDeserializers_IntegerDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      NumberDeserializers.NumberDeserializer numberDeserializers_NumberDeserializer0 = new NumberDeserializers.NumberDeserializer();
      assertFalse(numberDeserializers_NumberDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      NumberDeserializers numberDeserializers0 = new NumberDeserializers();
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Class<Character> class0 = Character.class;
      Character character0 = new Character('M');
      NumberDeserializers.CharacterDeserializer numberDeserializers_CharacterDeserializer0 = new NumberDeserializers.CharacterDeserializer(class0, character0);
      assertFalse(numberDeserializers_CharacterDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<Short> class0 = Short.class;
      NumberDeserializers.ShortDeserializer numberDeserializers_ShortDeserializer0 = new NumberDeserializers.ShortDeserializer(class0, (Short) null);
      assertFalse(numberDeserializers_ShortDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(arrayNode0);
      Class<Short> class0 = Short.class;
      try { 
        objectMapper0.readValue(jsonParser0, class0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not deserialize instance of java.lang.Short out of START_ARRAY token
         //  at [Source: java.lang.String@0000000010; line: -1, column: -1]
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<Byte> class0 = Byte.class;
      Byte byte0 = new Byte((byte) (-71));
      NumberDeserializers.ByteDeserializer numberDeserializers_ByteDeserializer0 = new NumberDeserializers.ByteDeserializer(class0, byte0);
      assertFalse(numberDeserializers_ByteDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      NumberDeserializers.ByteDeserializer numberDeserializers_ByteDeserializer0 = NumberDeserializers.ByteDeserializer.primitiveInstance;
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayList<JsonNode> arrayList0 = new ArrayList<JsonNode>();
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0, arrayList0);
      JsonParser jsonParser0 = arrayNode0.traverse();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      try { 
        numberDeserializers_ByteDeserializer0.deserialize(jsonParser0, (DeserializationContext) defaultDeserializationContext_Impl0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not deserialize instance of byte out of null token
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      FloatNode floatNode0 = new FloatNode(0.0F);
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(floatNode0);
      NumberDeserializers.BooleanDeserializer numberDeserializers_BooleanDeserializer0 = NumberDeserializers.BooleanDeserializer.primitiveInstance;
      // Undeclared exception!
      try { 
        numberDeserializers_BooleanDeserializer0.deserializeWithType(jsonParser0, (DeserializationContext) null, (TypeDeserializer) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<Boolean> class0 = Boolean.class;
      NumberDeserializers.BooleanDeserializer numberDeserializers_BooleanDeserializer0 = new NumberDeserializers.BooleanDeserializer(class0, (Boolean) null);
      assertFalse(numberDeserializers_BooleanDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0, defaultSerializerProvider_Impl0, (DefaultDeserializationContext) null);
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      Class<Boolean> class0 = Boolean.class;
      try { 
        objectMapper0.treeToValue((TreeNode) arrayNode0, class0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not deserialize instance of java.lang.Boolean out of START_ARRAY token
         //  at [Source: java.lang.String@0000000010; line: -1, column: -1]
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Class<Float> class0 = Float.class;
      Float float0 = new Float(1.0F);
      NumberDeserializers.FloatDeserializer numberDeserializers_FloatDeserializer0 = new NumberDeserializers.FloatDeserializer(class0, float0);
      assertFalse(numberDeserializers_FloatDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      NumberDeserializers.FloatDeserializer numberDeserializers_FloatDeserializer0 = NumberDeserializers.FloatDeserializer.wrapperInstance;
      JsonFactory jsonFactory0 = new JsonFactory();
      StringReader stringReader0 = new StringReader("");
      JsonParser jsonParser0 = jsonFactory0.createParser((Reader) stringReader0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      try { 
        numberDeserializers_FloatDeserializer0.deserialize(jsonParser0, (DeserializationContext) defaultDeserializationContext_Impl0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not deserialize instance of java.lang.Float out of null token
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Class<Double> class0 = Double.class;
      Double double0 = new Double(0.0);
      NumberDeserializers.DoubleDeserializer numberDeserializers_DoubleDeserializer0 = new NumberDeserializers.DoubleDeserializer(class0, double0);
      assertFalse(numberDeserializers_DoubleDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      NumberDeserializers.DoubleDeserializer numberDeserializers_DoubleDeserializer0 = NumberDeserializers.DoubleDeserializer.wrapperInstance;
      JsonFactory jsonFactory0 = new JsonFactory();
      byte[] byteArray0 = new byte[0];
      JsonParser jsonParser0 = jsonFactory0.createParser(byteArray0, 7, 0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      try { 
        numberDeserializers_DoubleDeserializer0.deserialize(jsonParser0, (DeserializationContext) defaultDeserializationContext_Impl0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not deserialize instance of java.lang.Double out of null token
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      NumberDeserializers.DoubleDeserializer numberDeserializers_DoubleDeserializer0 = NumberDeserializers.DoubleDeserializer.wrapperInstance;
      JsonFactory jsonFactory0 = new JsonFactory();
      char[] charArray0 = new char[6];
      JsonParser jsonParser0 = jsonFactory0.createParser(charArray0);
      try { 
        numberDeserializers_DoubleDeserializer0.deserializeWithType(jsonParser0, defaultDeserializationContext_Impl0, (TypeDeserializer) null);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not deserialize instance of java.lang.Double out of null token
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Class<Byte> class0 = Byte.TYPE;
      JsonDeserializer<?> jsonDeserializer0 = NumberDeserializers.find(class0, "r-E(v");
      assertNotNull(jsonDeserializer0);
      assertFalse(jsonDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Class<Integer> class0 = Integer.TYPE;
      JsonDeserializer<?> jsonDeserializer0 = NumberDeserializers.find(class0, "l#h=bce'x[$wfK(>");
      assertTrue(jsonDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Class<Boolean> class0 = Boolean.TYPE;
      JsonDeserializer<?> jsonDeserializer0 = NumberDeserializers.find(class0, "com.fasterxml.jackson.databind.deser.std.NumberDeserializers$PrimitiveOrWrapperDeserializer");
      assertFalse(jsonDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Class<Long> class0 = Long.TYPE;
      JsonDeserializer<?> jsonDeserializer0 = NumberDeserializers.find(class0, "x:");
      assertTrue(jsonDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Class<Double> class0 = Double.TYPE;
      JsonDeserializer<?> jsonDeserializer0 = NumberDeserializers.find(class0, (String) null);
      assertFalse(jsonDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Class<Character> class0 = Character.TYPE;
      JsonDeserializer<?> jsonDeserializer0 = NumberDeserializers.find(class0, "");
      assertFalse(jsonDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Class<Float> class0 = Float.TYPE;
      JsonDeserializer<?> jsonDeserializer0 = NumberDeserializers.find(class0, "@]f6V N}YJHPkc;[");
      assertFalse(jsonDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Class<Short> class0 = Short.TYPE;
      JsonDeserializer<?> jsonDeserializer0 = NumberDeserializers.find(class0, "|BgR*qK");
      assertFalse(jsonDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Class<String> class0 = String.class;
      JsonDeserializer<?> jsonDeserializer0 = NumberDeserializers.find(class0, "com.fasterxml.jackson.databnd.deser.td.NumberDeserializers$ByteDeseralizer");
      assertNull(jsonDeserializer0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<Double> class0 = Double.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<Character> class0 = Character.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Class<Byte> class0 = Byte.class;
      JsonDeserializer<?> jsonDeserializer0 = NumberDeserializers.find(class0, "java.math.BigInteger");
      assertFalse(jsonDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<BigInteger> class0 = BigInteger.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<Float> class0 = Float.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      NumberDeserializers.IntegerDeserializer numberDeserializers_IntegerDeserializer0 = NumberDeserializers.IntegerDeserializer.wrapperInstance;
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(numberDeserializers_IntegerDeserializer0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<BigDecimal> class0 = BigDecimal.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Class<BufferedInputStream> class0 = BufferedInputStream.class;
      // Undeclared exception!
      try { 
        NumberDeserializers.find(class0, "java.math.BigInteger");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Internal error: can't find deserializer for java.io.BufferedInputStream
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.NumberDeserializers", e);
      }
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      NumberDeserializers.IntegerDeserializer numberDeserializers_IntegerDeserializer0 = NumberDeserializers.IntegerDeserializer.wrapperInstance;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Integer integer0 = numberDeserializers_IntegerDeserializer0.getNullValue((DeserializationContext) defaultDeserializationContext_Impl0);
      assertNull(integer0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      NumberDeserializers.IntegerDeserializer numberDeserializers_IntegerDeserializer0 = NumberDeserializers.IntegerDeserializer.primitiveInstance;
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Integer integer0 = numberDeserializers_IntegerDeserializer0.getNullValue((DeserializationContext) defaultDeserializationContext_Impl0);
      assertEquals(0, (int)integer0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser("r-E(v");
      NumberDeserializers.CharacterDeserializer numberDeserializers_CharacterDeserializer0 = NumberDeserializers.CharacterDeserializer.primitiveInstance;
      // Undeclared exception!
      try { 
        numberDeserializers_CharacterDeserializer0.deserialize(jsonParser0, (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.NumberDeserializers$CharacterDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser("@]f6V N}YJHPkc;[");
      NumberDeserializers.IntegerDeserializer numberDeserializers_IntegerDeserializer0 = NumberDeserializers.IntegerDeserializer.primitiveInstance;
      // Undeclared exception!
      try { 
        numberDeserializers_IntegerDeserializer0.deserialize(jsonParser0, (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser("@]f6V N}YJHPkc;[");
      NumberDeserializers.LongDeserializer numberDeserializers_LongDeserializer0 = NumberDeserializers.LongDeserializer.wrapperInstance;
      // Undeclared exception!
      try { 
        numberDeserializers_LongDeserializer0.deserialize(jsonParser0, (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      NumberDeserializers.NumberDeserializer numberDeserializers_NumberDeserializer0 = NumberDeserializers.NumberDeserializer.instance;
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser("r-E(v");
      // Undeclared exception!
      try { 
        numberDeserializers_NumberDeserializer0.deserialize(jsonParser0, (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.NumberDeserializers$NumberDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(arrayNode0);
      NumberDeserializers.NumberDeserializer numberDeserializers_NumberDeserializer0 = NumberDeserializers.NumberDeserializer.instance;
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      // Undeclared exception!
      try { 
        numberDeserializers_NumberDeserializer0.deserializeWithType(jsonParser0, deserializationContext0, (TypeDeserializer) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.NumberDeserializers$NumberDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser("com.fasterxml.jackson.databnd.deser.td.NumberDeserializers$ByteDeseralizer");
      NumberDeserializers.BigIntegerDeserializer numberDeserializers_BigIntegerDeserializer0 = new NumberDeserializers.BigIntegerDeserializer();
      // Undeclared exception!
      try { 
        numberDeserializers_BigIntegerDeserializer0.deserialize(jsonParser0, (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.NumberDeserializers$BigIntegerDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser("r-E(v");
      NumberDeserializers.BigDecimalDeserializer numberDeserializers_BigDecimalDeserializer0 = NumberDeserializers.BigDecimalDeserializer.instance;
      // Undeclared exception!
      try { 
        numberDeserializers_BigDecimalDeserializer0.deserialize(jsonParser0, (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.NumberDeserializers$BigDecimalDeserializer", e);
      }
  }
}
