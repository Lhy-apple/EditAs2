/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:56:39 GMT 2023
 */

package com.fasterxml.jackson.databind;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.MappingIterator;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import java.io.IOException;
import java.io.InputStream;
import java.sql.SQLInvalidAuthorizationSpecException;
import java.util.Collection;
import java.util.LinkedList;
import java.util.NoSuchElementException;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class MappingIterator_ESTest extends MappingIterator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory((ObjectCodec) null);
      ObjectMapper objectMapper0 = new ObjectMapper();
      char[] charArray0 = new char[0];
      JsonParser jsonParser0 = jsonFactory0.createParser(charArray0);
      Class<ObjectMapper.DefaultTyping> class0 = ObjectMapper.DefaultTyping.class;
      MappingIterator<ObjectMapper.DefaultTyping> mappingIterator0 = objectMapper0.readValues(jsonParser0, class0);
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
      MappingIterator<ObjectMapper.DefaultTyping> mappingIterator0 = MappingIterator.emptyIterator();
      // Undeclared exception!
      try { 
        mappingIterator0._handleIOException((IOException) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.MappingIterator", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      MappingIterator<ObjectMapper.DefaultTyping> mappingIterator0 = MappingIterator.emptyIterator();
      boolean boolean0 = mappingIterator0.hasNext();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      JsonFactory jsonFactory0 = new JsonFactory((ObjectCodec) null);
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0, (DefaultSerializerProvider) null, defaultDeserializationContext_Impl0);
      MappingIterator<Boolean> mappingIterator0 = objectMapper0.readValues((JsonParser) null, (JavaType) simpleType0);
      JsonParser jsonParser0 = mappingIterator0.getParser();
      assertNull(jsonParser0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      MappingIterator<ObjectMapper.DefaultTyping> mappingIterator0 = MappingIterator.emptyIterator();
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
      MappingIterator<ObjectMapper.DefaultTyping> mappingIterator0 = MappingIterator.emptyIterator();
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
      MappingIterator<String> mappingIterator0 = MappingIterator.emptyIterator();
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
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(false);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      ArrayNode arrayNode0 = objectNode0.putArray("EFN6O}");
      JsonParser jsonParser0 = arrayNode0.traverse((ObjectCodec) objectMapper0);
      Class<ObjectMapper.DefaultTyping> class0 = ObjectMapper.DefaultTyping.class;
      MappingIterator<ObjectMapper.DefaultTyping> mappingIterator0 = objectMapper0.readValues(jsonParser0, class0);
      SQLInvalidAuthorizationSpecException sQLInvalidAuthorizationSpecException0 = new SQLInvalidAuthorizationSpecException("=Qg)vZw_&rK|qD");
      JsonMappingException jsonMappingException0 = new JsonMappingException("=Qg)vZw_&rK|qD", sQLInvalidAuthorizationSpecException0);
      // Undeclared exception!
      try { 
        mappingIterator0._handleMappingException(jsonMappingException0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // =Qg)vZw_&rK|qD
         //
         verifyException("com.fasterxml.jackson.databind.MappingIterator", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonDeserializer<String> jsonDeserializer0 = (JsonDeserializer<String>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      MappingIterator<Object> mappingIterator0 = new MappingIterator<Object>((JavaType) null, (JsonParser) null, defaultDeserializationContext_Impl0, jsonDeserializer0, true, jsonDeserializer0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory((ObjectCodec) null);
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0, (DefaultSerializerProvider) null, defaultDeserializationContext_Impl0);
      Class<InputStream> class0 = InputStream.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      byte[] byteArray0 = new byte[7];
      byteArray0[0] = (byte)91;
      byteArray0[2] = (byte)125;
      MappingIterator<Object> mappingIterator0 = objectReader0.readValues(byteArray0);
      assertNotNull(mappingIterator0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      MappingIterator<ObjectMapper.DefaultTyping> mappingIterator0 = MappingIterator.emptyIterator();
      mappingIterator0.close();
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectNode objectNode0 = objectMapper0.createObjectNode();
      ArrayNode arrayNode0 = objectNode0.putArray("EFN6O}");
      JsonParser jsonParser0 = arrayNode0.traverse((ObjectCodec) objectMapper0);
      Class<ObjectMapper.DefaultTyping> class0 = ObjectMapper.DefaultTyping.class;
      MappingIterator<ObjectMapper.DefaultTyping> mappingIterator0 = objectMapper0.readValues(jsonParser0, class0);
      mappingIterator0.close();
      assertTrue(jsonParser0.isClosed());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse((ObjectCodec) objectMapper0);
      Class<ObjectMapper.DefaultTyping> class0 = ObjectMapper.DefaultTyping.class;
      MappingIterator<ObjectMapper.DefaultTyping> mappingIterator0 = objectMapper0.readValues(jsonParser0, class0);
      mappingIterator0.hasNextValue();
      try { 
        mappingIterator0.readAll();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not deserialize instance of com.fasterxml.jackson.databind.ObjectMapper$DefaultTyping out of START_ARRAY token
         //  at [Source: java.lang.String@0000000116; line: -1, column: -1]
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(false);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      ArrayNode arrayNode0 = objectNode0.putArray("v/7b27!g5%k]?kPm%");
      JsonParser jsonParser0 = arrayNode0.traverse((ObjectCodec) objectMapper0);
      Class<ObjectMapper.DefaultTyping> class0 = ObjectMapper.DefaultTyping.class;
      MappingIterator<ObjectMapper.DefaultTyping> mappingIterator0 = objectMapper0.readValues(jsonParser0, class0);
      Class<Integer> class1 = Integer.class;
      MappingIterator<Integer> mappingIterator1 = objectMapper0.readValues(jsonParser0, class1);
      mappingIterator1.hasNextValue();
      try { 
        mappingIterator0.readAll();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not deserialize instance of com.fasterxml.jackson.databind.ObjectMapper$DefaultTyping out of START_ARRAY token
         //  at [Source: java.lang.String@0000000116; line: -1, column: -1]
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(true);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      ArrayNode arrayNode0 = objectNode0.putArray("v/7b27!g5%k]?kPm%");
      JsonParser jsonParser0 = arrayNode0.traverse((ObjectCodec) objectMapper0);
      Class<ObjectMapper.DefaultTyping> class0 = ObjectMapper.DefaultTyping.class;
      MappingIterator<ObjectMapper.DefaultTyping> mappingIterator0 = objectMapper0.readValues(jsonParser0, class0);
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonDeserializer<InputStream> jsonDeserializer0 = (JsonDeserializer<InputStream>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      doReturn((Object) null).when(jsonDeserializer0).deserialize(any(com.fasterxml.jackson.core.JsonParser.class) , any(com.fasterxml.jackson.databind.DeserializationContext.class) , any(java.io.InputStream.class));
      MappingIterator<ObjectMapper.DefaultTyping> mappingIterator1 = new MappingIterator<ObjectMapper.DefaultTyping>(simpleType0, jsonParser0, defaultDeserializationContext_Impl0, jsonDeserializer0, true, mappingIterator0);
      mappingIterator1.readAll();
      assertTrue(jsonParser0.isClosed());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(true);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      ArrayNode arrayNode0 = objectNode0.putArray((String) null);
      JsonParser jsonParser0 = arrayNode0.traverse((ObjectCodec) objectMapper0);
      Class<ObjectMapper.DefaultTyping> class0 = ObjectMapper.DefaultTyping.class;
      MappingIterator<ObjectMapper.DefaultTyping> mappingIterator0 = objectMapper0.readValues(jsonParser0, class0);
      try { 
        mappingIterator0.nextValue();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not deserialize instance of com.fasterxml.jackson.databind.ObjectMapper$DefaultTyping out of START_ARRAY token
         //  at [Source: java.lang.String@0000000116; line: -1, column: -1]
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory((ObjectCodec) null);
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0, (DefaultSerializerProvider) null, defaultDeserializationContext_Impl0);
      Class<InputStream> class0 = InputStream.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      byte[] byteArray0 = new byte[0];
      MappingIterator<Boolean> mappingIterator0 = objectReader0.readValues(byteArray0);
      mappingIterator0.hasNextValue();
      // Undeclared exception!
      try { 
        mappingIterator0.nextValue();
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.MappingIterator", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      MappingIterator<Object> mappingIterator0 = MappingIterator.emptyIterator();
      Collection<ObjectReader> collection0 = mappingIterator0.readAll((Collection<ObjectReader>) null);
      assertNull(collection0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(true);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      ArrayNode arrayNode0 = objectNode0.putArray("v/7b27!g5%k]?kPm%");
      JsonParser jsonParser0 = arrayNode0.traverse((ObjectCodec) objectMapper0);
      Class<ObjectMapper.DefaultTyping> class0 = ObjectMapper.DefaultTyping.class;
      MappingIterator<ObjectMapper.DefaultTyping> mappingIterator0 = objectMapper0.readValues(jsonParser0, class0);
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      try { 
        mappingIterator0.readAll(linkedList0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not deserialize instance of com.fasterxml.jackson.databind.ObjectMapper$DefaultTyping out of START_ARRAY token
         //  at [Source: java.lang.String@0000000116; line: -1, column: -1]
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }
}