/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:27:20 GMT 2023
 */

package com.fasterxml.jackson.databind;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.core.util.JsonParserDelegate;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.MappingIterator;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ShortNode;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.type.ArrayType;
import com.fasterxml.jackson.databind.type.CollectionLikeType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.NoSuchElementException;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class MappingIterator_ESTest extends MappingIterator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      MappingIterator<Object> mappingIterator0 = MappingIterator.emptyIterator();
      // Undeclared exception!
      try { 
        mappingIterator0.next();
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.MappingIterator", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      MappingIterator<Integer> mappingIterator0 = MappingIterator.emptyIterator();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Object[] objectArray0 = new Object[0];
      JsonMappingException jsonMappingException0 = defaultSerializerProvider_Impl0.mappingException("", objectArray0);
      // Undeclared exception!
      try { 
        mappingIterator0._handleIOException((IOException) jsonMappingException0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // 
         //
         verifyException("com.fasterxml.jackson.databind.MappingIterator", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      MappingIterator<List<Integer>> mappingIterator0 = MappingIterator.emptyIterator();
      boolean boolean0 = mappingIterator0.hasNext();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(jsonNodeFactory0);
      File file0 = MockFile.createTempFile("V!sp)7q!v?u;", "V!sp)7q!v?u;");
      MappingIterator<Object> mappingIterator0 = objectReader0.readValues(file0);
      JsonParser jsonParser0 = mappingIterator0.getParser();
      assertNull(jsonParser0.getCurrentName());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      MappingIterator<String> mappingIterator0 = MappingIterator.emptyIterator();
      // Undeclared exception!
      try { 
        mappingIterator0.getCurrentLocation();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.MappingIterator", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      MappingIterator<List<JsonFactory.Feature>> mappingIterator0 = MappingIterator.emptyIterator();
      // Undeclared exception!
      try { 
        mappingIterator0.remove();
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.MappingIterator", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      MappingIterator<Boolean> mappingIterator0 = MappingIterator.emptyIterator();
      // Undeclared exception!
      try { 
        mappingIterator0.getParserSchema();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.MappingIterator", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      MappingIterator<List<Integer>> mappingIterator0 = MappingIterator.emptyIterator();
      // Undeclared exception!
      try { 
        mappingIterator0._handleMappingException((JsonMappingException) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.MappingIterator", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<InputStream> class0 = InputStream.class;
      JavaType javaType0 = TypeFactory.unknownType();
      CollectionLikeType collectionLikeType0 = CollectionLikeType.construct(class0, javaType0);
      ArrayType arrayType0 = ArrayType.construct(collectionLikeType0, javaType0, class0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonDeserializer<Boolean> jsonDeserializer0 = (JsonDeserializer<Boolean>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      MappingIterator<Boolean> mappingIterator0 = new MappingIterator<Boolean>(arrayType0, (JsonParser) null, defaultDeserializationContext_Impl0, jsonDeserializer0, true, arrayType0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      JsonParser jsonParser0 = arrayNode0.traverse();
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(jsonNodeFactory0);
      MappingIterator<Integer> mappingIterator0 = objectReader0._bindAndReadValues(jsonParser0, (Object) objectMapper0);
      mappingIterator0.readAll();
      assertTrue(jsonParser0.isClosed());
      assertEquals(JsonToken.START_ARRAY, jsonParser0.getLastClearedToken());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      MappingIterator<Integer> mappingIterator0 = MappingIterator.emptyIterator();
      mappingIterator0.close();
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(jsonNodeFactory0);
      byte[] byteArray0 = new byte[0];
      MappingIterator<Integer> mappingIterator0 = objectReader0.readValues(byteArray0, 27, 27);
      mappingIterator0.close();
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(jsonNodeFactory0);
      byte[] byteArray0 = new byte[0];
      MappingIterator<Integer> mappingIterator0 = objectReader0.readValues(byteArray0, 17, 17);
      mappingIterator0._hasNextChecked = true;
      List<Integer> list0 = mappingIterator0.readAll();
      assertFalse(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse();
      JsonParserDelegate jsonParserDelegate0 = new JsonParserDelegate(jsonParser0);
      Class<ShortNode> class0 = ShortNode.class;
      MappingIterator<ShortNode> mappingIterator0 = objectMapper0.readValues((JsonParser) jsonParserDelegate0, class0);
      mappingIterator0.hasNextValue();
      Class<Integer> class1 = Integer.class;
      MappingIterator<Integer> mappingIterator1 = objectMapper0.readValues(jsonParser0, class1);
      boolean boolean0 = mappingIterator1.hasNextValue();
      assertTrue(jsonParser0.hasCurrentToken());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse();
      Class<ShortNode> class0 = ShortNode.class;
      MappingIterator<ShortNode> mappingIterator0 = objectMapper0.readValues(jsonParser0, class0);
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      mappingIterator0.readAll((List<Object>) linkedList0);
      assertEquals(1, linkedList0.size());
      assertEquals(0, jsonParser0.getCurrentTokenId());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      JsonParser jsonParser0 = arrayNode0.traverse();
      Class<InputStream> class0 = InputStream.class;
      MappingIterator<InputStream> mappingIterator0 = objectMapper0.readValues(jsonParser0, class0);
      try { 
        mappingIterator0.nextValue();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not construct instance of java.io.InputStream, problem: abstract types either need to be mapped to concrete types, have custom deserializer, or be instantiated with additional type information
         //  at [Source: java.lang.String@0000000525; line: -1, column: -1]
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      JsonParser jsonParser0 = arrayNode0.traverse();
      JsonParserDelegate jsonParserDelegate0 = new JsonParserDelegate(jsonParser0);
      Class<ShortNode> class0 = ShortNode.class;
      MappingIterator<ShortNode> mappingIterator0 = objectMapper0.readValues((JsonParser) jsonParserDelegate0, class0);
      ArrayList<Object> arrayList0 = new ArrayList<Object>();
      mappingIterator0.readAll((List<Object>) arrayList0);
      assertEquals(1, arrayList0.size());
      assertFalse(jsonParser0.isExpectedStartArrayToken());
  }
}
