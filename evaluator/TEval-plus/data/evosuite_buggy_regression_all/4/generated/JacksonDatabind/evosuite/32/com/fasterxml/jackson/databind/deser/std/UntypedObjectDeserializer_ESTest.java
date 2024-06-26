/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:41:28 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.io.IOContext;
import com.fasterxml.jackson.core.json.ReaderBasedJsonParser;
import com.fasterxml.jackson.core.sym.CharsToNameCanonicalizer;
import com.fasterxml.jackson.core.util.BufferRecycler;
import com.fasterxml.jackson.core.util.JsonParserSequence;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.std.UntypedObjectDeserializer;
import com.fasterxml.jackson.databind.jsontype.TypeDeserializer;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.FloatNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.StringReader;
import java.net.MalformedURLException;
import java.net.URL;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.net.MockURL;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class UntypedObjectDeserializer_ESTest extends UntypedObjectDeserializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = new UntypedObjectDeserializer.Vanilla();
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(untypedObjectDeserializer_Vanilla0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = UntypedObjectDeserializer.Vanilla.std;
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      JsonDeserializer<?> jsonDeserializer0 = untypedObjectDeserializer0.instance._withResolved(untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer_Vanilla0);
      assertTrue(jsonDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      JsonDeserializer<Object> jsonDeserializer0 = untypedObjectDeserializer0._clearIfStdImpl((JsonDeserializer<Object>) null);
      assertNull(jsonDeserializer0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      JsonDeserializer<?> jsonDeserializer0 = untypedObjectDeserializer0.createContextual(defaultDeserializationContext_Impl0, (BeanProperty) null);
      assertFalse(jsonDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      JsonNodeFactory.withExactBigDecimals(false);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl1 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      UntypedObjectDeserializer untypedObjectDeserializer1 = new UntypedObjectDeserializer();
      untypedObjectDeserializer1.getNullValue((DeserializationContext) defaultDeserializationContext_Impl1);
      untypedObjectDeserializer0._listDeserializer = (JsonDeserializer<Object>) untypedObjectDeserializer1;
      untypedObjectDeserializer0.createContextual(defaultDeserializationContext_Impl0, (BeanProperty) null);
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = UntypedObjectDeserializer.Vanilla.std;
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonGenerator.Feature jsonGenerator_Feature0 = JsonGenerator.Feature.AUTO_CLOSE_JSON_CONTENT;
      jsonFactory0.disable(jsonGenerator_Feature0);
      try { 
        MockURL.URL("'gPMxpw^KO3~H");
        fail("Expecting exception: MalformedURLException");
      
      } catch(MalformedURLException e) {
         //
         // no protocol: 'gPMxpw^KO3~H
         //
         verifyException("java.net.URL", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      ObjectNode objectNode0 = arrayNode0.addObject();
      objectNode0.put((String) null, (-2764L));
      JsonParser jsonParser0 = arrayNode0.traverse();
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
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
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      UntypedObjectDeserializer untypedObjectDeserializer0 = new UntypedObjectDeserializer();
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      ObjectNode objectNode0 = arrayNode0.addObject();
      objectNode0.put((String) null, 0.0);
      objectNode0.put("L^onLs", "L^onLs");
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
  public void test07()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      ArrayNode arrayNode1 = arrayNode0.insert(6, false);
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      JsonParser jsonParser0 = arrayNode1.traverse();
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
  public void test08()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      ObjectMapper objectMapper0 = new ObjectMapper();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      UntypedObjectDeserializer untypedObjectDeserializer0 = new UntypedObjectDeserializer();
      arrayNode0.insertPOJO((-3365), (Object) null);
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
  public void test09()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      Double double0 = new Double(3);
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      ArrayNode arrayNode1 = arrayNode0.insertPOJO(4648, double0);
      JsonParser jsonParser0 = arrayNode1.traverse();
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
  public void test10()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      UntypedObjectDeserializer untypedObjectDeserializer0 = new UntypedObjectDeserializer();
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      ObjectNode objectNode0 = arrayNode0.addObject();
      objectNode0.put("$FFqKOh93", "$FFqKOh93");
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
  public void test11()  throws Throwable  {
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = UntypedObjectDeserializer.Vanilla.std;
      UntypedObjectDeserializer untypedObjectDeserializer0 = new UntypedObjectDeserializer();
      UntypedObjectDeserializer untypedObjectDeserializer1 = new UntypedObjectDeserializer(untypedObjectDeserializer0, untypedObjectDeserializer0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer0, untypedObjectDeserializer_Vanilla0);
      ObjectMapper objectMapper0 = new ObjectMapper();
      FloatNode floatNode0 = new FloatNode(0.0F);
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(floatNode0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
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
  public void test12()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = new UntypedObjectDeserializer();
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser("JSON");
      // Undeclared exception!
      try { 
        untypedObjectDeserializer0.deserializeWithType(jsonParser0, (DeserializationContext) null, (TypeDeserializer) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.UntypedObjectDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse();
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
  public void test14()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      arrayNode0.addObject();
      Double double0 = new Double(3);
      arrayNode0.add(double0);
      JsonParser jsonParser0 = arrayNode0.traverse();
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
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
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      arrayNode0.addObject();
      Double double0 = new Double(3);
      arrayNode0.insertObject(3);
      ArrayNode arrayNode1 = arrayNode0.add(double0);
      arrayNode1.insert(6, true);
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      JsonParser jsonParser0 = arrayNode1.traverse();
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
  public void test16()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = new UntypedObjectDeserializer();
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser("JSON");
      // Undeclared exception!
      try { 
        untypedObjectDeserializer0.mapObject(jsonParser0, (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.UntypedObjectDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      ObjectNode objectNode0 = arrayNode0.insertObject(4);
      ObjectNode objectNode1 = objectNode0.put("", 721.71);
      Long long0 = new Long(4);
      objectNode1.put("com.fasterxml.jackson.databind.util.NameTransformer$2", long0);
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      JsonParser jsonParser0 = arrayNode0.traverse();
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
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
  public void test18()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      ObjectNode objectNode0 = arrayNode0.insertObject(4);
      ObjectNode objectNode1 = objectNode0.put("", 721.71);
      Long long0 = new Long(4);
      ObjectNode objectNode2 = objectNode1.put("com.fasterxml.jackson.databind.util.NameTransformer$2", long0);
      Float float0 = new Float((-180.335964604));
      ObjectNode objectNode3 = objectNode2.put("com.fasterxml.jackson.databind.deser.std.UntypedObjectDeserializer", float0);
      objectNode3.put("@Bb}0V]oA", "com.fasterxml.jackson.databind.deser.std.UntypedObjectDeserializer");
      JsonParser jsonParser0 = arrayNode0.traverse();
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
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
  public void test19()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      ArrayNode arrayNode0 = objectNode0.putArray("JSON");
      JsonParser jsonParser0 = arrayNode0.traverse();
      JsonParserSequence jsonParserSequence0 = JsonParserSequence.createFlattened(jsonParser0, jsonParser0);
      jsonParserSequence0.nextValue();
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Object[] objectArray0 = untypedObjectDeserializer0.mapArrayToArray(jsonParserSequence0, defaultDeserializationContext_Impl0);
      assertEquals(0, objectArray0.length);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      UntypedObjectDeserializer untypedObjectDeserializer0 = new UntypedObjectDeserializer();
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      arrayNode0.addObject();
      JsonParser jsonParser0 = arrayNode0.traverse();
      JsonParserSequence jsonParserSequence0 = JsonParserSequence.createFlattened(jsonParser0, jsonParser0);
      jsonParserSequence0.nextToken();
      Object[] objectArray0 = untypedObjectDeserializer0.mapArrayToArray(jsonParserSequence0, defaultDeserializationContext_Impl0);
      assertEquals(1, objectArray0.length);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser("JSON");
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = new UntypedObjectDeserializer.Vanilla();
      // Undeclared exception!
      try { 
        untypedObjectDeserializer_Vanilla0.deserialize(jsonParser0, (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.UntypedObjectDeserializer$Vanilla", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = new UntypedObjectDeserializer.Vanilla();
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      Double double0 = new Double(0.0);
      IOContext iOContext0 = new IOContext(bufferRecycler0, double0, false);
      StringReader stringReader0 = new StringReader("qu{ +/;djoh[R?'1,%6");
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 0, stringReader0, objectMapper0, charsToNameCanonicalizer0);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      // Undeclared exception!
      try { 
        untypedObjectDeserializer_Vanilla0.deserializeWithType(readerBasedJsonParser0, deserializationContext0, (TypeDeserializer) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }
}
