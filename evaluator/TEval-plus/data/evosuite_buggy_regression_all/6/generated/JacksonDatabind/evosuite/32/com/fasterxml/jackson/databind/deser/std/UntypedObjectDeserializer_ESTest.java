/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:23:52 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.core.util.JsonParserDelegate;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.std.UntypedObjectDeserializer;
import com.fasterxml.jackson.databind.jsontype.TypeDeserializer;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.BooleanNode;
import com.fasterxml.jackson.databind.node.DecimalNode;
import com.fasterxml.jackson.databind.node.FloatNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.node.TextNode;
import com.fasterxml.jackson.databind.util.RawValue;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.math.BigDecimal;
import java.util.HashMap;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class UntypedObjectDeserializer_ESTest extends UntypedObjectDeserializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = new UntypedObjectDeserializer.Vanilla();
      UntypedObjectDeserializer untypedObjectDeserializer0 = new UntypedObjectDeserializer();
      JsonDeserializer<?> jsonDeserializer0 = untypedObjectDeserializer0.instance._withResolved(untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer_Vanilla0);
      assertNotSame(untypedObjectDeserializer0, jsonDeserializer0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = new UntypedObjectDeserializer();
      JsonDeserializer<Object> jsonDeserializer0 = untypedObjectDeserializer0.instance._clearIfStdImpl((JsonDeserializer<Object>) null);
      assertNull(jsonDeserializer0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = UntypedObjectDeserializer.Vanilla.std;
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(untypedObjectDeserializer_Vanilla0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = new UntypedObjectDeserializer.Vanilla();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      UntypedObjectDeserializer untypedObjectDeserializer1 = new UntypedObjectDeserializer(untypedObjectDeserializer0, untypedObjectDeserializer0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer0);
      JsonDeserializer<?> jsonDeserializer0 = untypedObjectDeserializer1.createContextual(defaultDeserializationContext_Impl0, (BeanProperty) null);
      assertSame(untypedObjectDeserializer1, jsonDeserializer0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      ObjectMapper objectMapper0 = new ObjectMapper();
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      JsonDeserializer<?> jsonDeserializer0 = untypedObjectDeserializer0.createContextual(deserializationContext0, (BeanProperty) null);
      assertFalse(jsonDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse();
      // Undeclared exception!
      try { 
        untypedObjectDeserializer0.mapArrayToArray(jsonParser0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = new UntypedObjectDeserializer();
      ObjectMapper objectMapper0 = new ObjectMapper();
      TextNode textNode0 = new TextNode("JSON");
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(textNode0);
      JsonParserDelegate jsonParserDelegate0 = new JsonParserDelegate(jsonParser0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        untypedObjectDeserializer0.mapArrayToArray(jsonParserDelegate0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = new UntypedObjectDeserializer();
      ObjectMapper objectMapper0 = new ObjectMapper();
      BooleanNode booleanNode0 = BooleanNode.valueOf(true);
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(booleanNode0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        untypedObjectDeserializer0.mapArray(jsonParser0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      Boolean boolean0 = Boolean.valueOf("");
      objectNode0.put("", boolean0);
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(objectNode0);
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      UntypedObjectDeserializer untypedObjectDeserializer0 = new UntypedObjectDeserializer();
      // Undeclared exception!
      try { 
        untypedObjectDeserializer0.mapArray(jsonParser0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      ObjectNode objectNode1 = objectNode0.put("JSON", (String) null);
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(objectNode1);
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      UntypedObjectDeserializer untypedObjectDeserializer0 = new UntypedObjectDeserializer();
      // Undeclared exception!
      try { 
        untypedObjectDeserializer0.mapArray(jsonParser0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = UntypedObjectDeserializer.Vanilla.std;
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse();
      UntypedObjectDeserializer untypedObjectDeserializer1 = new UntypedObjectDeserializer(untypedObjectDeserializer0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer0, untypedObjectDeserializer_Vanilla0);
      // Undeclared exception!
      try { 
        untypedObjectDeserializer1.mapArrayToArray(jsonParser0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      Integer integer0 = Integer.valueOf(0);
      arrayNode0.insert((-1608), integer0);
      JsonParser jsonParser0 = arrayNode0.traverse();
      // Undeclared exception!
      try { 
        untypedObjectDeserializer0.mapArrayToArray(jsonParser0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      DecimalNode decimalNode0 = DecimalNode.ZERO;
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(decimalNode0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        untypedObjectDeserializer0.deserializeWithType(jsonParser0, defaultDeserializationContext_Impl0, (TypeDeserializer) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(true);
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      Integer integer0 = new Integer(0);
      BigDecimal bigDecimal0 = new BigDecimal(0L);
      ArrayNode arrayNode1 = arrayNode0.add(bigDecimal0);
      arrayNode1.insert(0, integer0);
      JsonParser jsonParser0 = arrayNode1.traverse();
      UntypedObjectDeserializer untypedObjectDeserializer0 = new UntypedObjectDeserializer();
      // Undeclared exception!
      try { 
        untypedObjectDeserializer0.mapArrayToArray(jsonParser0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      Integer integer0 = new Integer(0);
      byte[] byteArray0 = new byte[1];
      arrayNode0.insert((int) (byte)0, byteArray0);
      BigDecimal bigDecimal0 = new BigDecimal((long) (byte)18);
      arrayNode0.add(bigDecimal0);
      ArrayNode arrayNode1 = arrayNode0.insert((int) (byte)18, integer0);
      JsonParser jsonParser0 = arrayNode1.traverse();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        untypedObjectDeserializer0.mapArrayToArray(jsonParser0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      ObjectMapper objectMapper0 = new ObjectMapper();
      DecimalNode decimalNode0 = DecimalNode.ZERO;
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(decimalNode0);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      try { 
        untypedObjectDeserializer0.mapObject(jsonParser0, deserializationContext0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not deserialize instance of java.lang.Object out of null token
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectNode objectNode0 = objectMapper0.createObjectNode();
      ObjectNode objectNode1 = objectNode0.put((String) null, (-444L));
      ObjectNode objectNode2 = objectNode1.put("JSON", 2047);
      objectNode1.replace("qNG^5{F Ng", objectNode2);
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(objectNode1);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      UntypedObjectDeserializer untypedObjectDeserializer0 = new UntypedObjectDeserializer();
      // Undeclared exception!
      untypedObjectDeserializer0.mapArray(jsonParser0, defaultDeserializationContext_Impl0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      objectNode0.with(".d~qHGa6|0>H?(x");
      FloatNode floatNode0 = FloatNode.valueOf(3574.7717F);
      BigDecimal bigDecimal0 = floatNode0.decimalValue();
      ObjectNode objectNode1 = objectNode0.put("", bigDecimal0);
      RawValue rawValue0 = new RawValue("");
      objectNode0.putRawValue("UUID has to be represented by the standard 36-char representation", rawValue0);
      Double double0 = new Double((-1096.5725946523983));
      objectNode0.put("aT,L50wG6cj5E", double0);
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(objectNode1);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      UntypedObjectDeserializer untypedObjectDeserializer0 = new UntypedObjectDeserializer();
      // Undeclared exception!
      try { 
        untypedObjectDeserializer0.mapArray(jsonParser0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      ObjectNode objectNode1 = objectNode0.put((String) null, (-444L));
      ObjectNode objectNode2 = objectNode1.put("?@=]7`diN/w=$X0}Rn", (-1884));
      hashMap0.replace("?@=]7`diN/w=$X0}Rn", (JsonNode) objectNode1);
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(objectNode2);
      UntypedObjectDeserializer untypedObjectDeserializer0 = new UntypedObjectDeserializer((JavaType) null, (JavaType) null);
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      untypedObjectDeserializer0.mapArrayToArray(jsonParser0, defaultDeserializationContext_Impl0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = UntypedObjectDeserializer.Vanilla.std;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse();
      UntypedObjectDeserializer untypedObjectDeserializer0 = new UntypedObjectDeserializer();
      JsonDeserializer<Double> jsonDeserializer0 = (JsonDeserializer<Double>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      doReturn((Object) null).when(jsonDeserializer0).deserialize(any(com.fasterxml.jackson.core.JsonParser.class) , any(com.fasterxml.jackson.databind.DeserializationContext.class));
      JsonDeserializer<ObjectInputStream> jsonDeserializer1 = (JsonDeserializer<ObjectInputStream>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      UntypedObjectDeserializer untypedObjectDeserializer1 = new UntypedObjectDeserializer(untypedObjectDeserializer0, untypedObjectDeserializer_Vanilla0, jsonDeserializer0, jsonDeserializer1, jsonDeserializer1);
      Object[] objectArray0 = untypedObjectDeserializer1.mapArrayToArray(jsonParser0, defaultDeserializationContext_Impl0);
      assertEquals(4, jsonParser0.getCurrentTokenId());
      assertEquals(1, objectArray0.length);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = new UntypedObjectDeserializer.Vanilla();
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      objectNode0.put("", (BigDecimal) null);
      objectNode0.put((String) null, 0.0F);
      ObjectNode objectNode1 = objectNode0.put("aT,L50wG6cj5E", (-374L));
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(objectNode1);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      UntypedObjectDeserializer untypedObjectDeserializer0 = new UntypedObjectDeserializer();
      UntypedObjectDeserializer untypedObjectDeserializer1 = new UntypedObjectDeserializer(untypedObjectDeserializer0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer0, untypedObjectDeserializer0, untypedObjectDeserializer0);
      // Undeclared exception!
      try { 
        untypedObjectDeserializer1.mapArray(jsonParser0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = UntypedObjectDeserializer.Vanilla.std;
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      BigDecimal bigDecimal0 = new BigDecimal(1667L);
      arrayNode0.add(bigDecimal0);
      JsonParser jsonParser0 = arrayNode0.traverse();
      UntypedObjectDeserializer untypedObjectDeserializer1 = new UntypedObjectDeserializer(untypedObjectDeserializer0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer0, untypedObjectDeserializer_Vanilla0);
      // Undeclared exception!
      try { 
        untypedObjectDeserializer1.mapArrayToArray(jsonParser0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = new UntypedObjectDeserializer.Vanilla();
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      RawValue rawValue0 = new RawValue((String) null);
      ObjectNode objectNode1 = objectNode0.putRawValue("", rawValue0);
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(objectNode1);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      UntypedObjectDeserializer untypedObjectDeserializer0 = new UntypedObjectDeserializer();
      UntypedObjectDeserializer untypedObjectDeserializer1 = new UntypedObjectDeserializer(untypedObjectDeserializer0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer0, untypedObjectDeserializer0, untypedObjectDeserializer0);
      // Undeclared exception!
      try { 
        untypedObjectDeserializer1.mapArray(jsonParser0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      DecimalNode decimalNode0 = DecimalNode.ZERO;
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(decimalNode0);
      JsonParserDelegate jsonParserDelegate0 = new JsonParserDelegate(jsonParser0);
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = new UntypedObjectDeserializer.Vanilla();
      // Undeclared exception!
      try { 
        untypedObjectDeserializer_Vanilla0.mapArray(jsonParserDelegate0, (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.UntypedObjectDeserializer$Vanilla", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse((ObjectCodec) objectMapper0);
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = new UntypedObjectDeserializer.Vanilla();
      // Undeclared exception!
      try { 
        untypedObjectDeserializer_Vanilla0.deserializeWithType(jsonParser0, deserializationContext0, (TypeDeserializer) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = UntypedObjectDeserializer.Vanilla.std;
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      arrayNode0.insert((int) (byte)18, (byte[]) null);
      BigDecimal bigDecimal0 = new BigDecimal(1667L);
      arrayNode0.add(bigDecimal0);
      JsonParser jsonParser0 = arrayNode0.traverse();
      UntypedObjectDeserializer untypedObjectDeserializer1 = new UntypedObjectDeserializer(untypedObjectDeserializer0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer0, untypedObjectDeserializer_Vanilla0);
      // Undeclared exception!
      try { 
        untypedObjectDeserializer1.mapArrayToArray(jsonParser0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = UntypedObjectDeserializer.Vanilla.std;
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      ArrayNode arrayNode1 = arrayNode0.insert((int) (byte)18, (byte[]) null);
      BigDecimal bigDecimal0 = new BigDecimal(1667L);
      arrayNode0.add(bigDecimal0);
      arrayNode1.insert(1519, (Integer) null);
      JsonParser jsonParser0 = arrayNode0.traverse();
      UntypedObjectDeserializer untypedObjectDeserializer1 = new UntypedObjectDeserializer(untypedObjectDeserializer0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer0, untypedObjectDeserializer_Vanilla0);
      // Undeclared exception!
      try { 
        untypedObjectDeserializer1.mapArrayToArray(jsonParser0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = new UntypedObjectDeserializer.Vanilla();
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      ObjectNode objectNode1 = objectNode0.put(".d~qHGa6|0>H?x", 1009L);
      objectNode1.put("KW=DhPas", false);
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(objectNode0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      UntypedObjectDeserializer untypedObjectDeserializer0 = new UntypedObjectDeserializer();
      UntypedObjectDeserializer untypedObjectDeserializer1 = new UntypedObjectDeserializer(untypedObjectDeserializer0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer_Vanilla0);
      // Undeclared exception!
      try { 
        untypedObjectDeserializer1.mapArray(jsonParser0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = new UntypedObjectDeserializer.Vanilla();
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      ObjectNode objectNode1 = objectNode0.with(".d~qHGa6|0>H?(x");
      objectNode0.put("JSON", 0L);
      hashMap0.put("com.fasterxml.jackson.databind.deser.std.UntypedObjectDeserializer$Vanilla", objectNode1);
      objectNode0.put("", 44);
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(objectNode0);
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      UntypedObjectDeserializer untypedObjectDeserializer1 = new UntypedObjectDeserializer(untypedObjectDeserializer0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer_Vanilla0);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      // Undeclared exception!
      try { 
        untypedObjectDeserializer1.mapArray(jsonParser0, deserializationContext0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }
}