/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 03:08:46 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.core.filter.FilteringParserDelegate;
import com.fasterxml.jackson.core.filter.TokenFilter;
import com.fasterxml.jackson.core.util.JsonParserSequence;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.std.NumberDeserializers;
import com.fasterxml.jackson.databind.jsontype.TypeDeserializer;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import java.io.IOException;
import java.io.InputStream;
import java.io.PushbackInputStream;
import java.io.SequenceInputStream;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.time.chrono.ChronoLocalDate;
import java.util.Enumeration;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class NumberDeserializers_ESTest extends NumberDeserializers_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      NumberDeserializers.BigIntegerDeserializer numberDeserializers_BigIntegerDeserializer0 = new NumberDeserializers.BigIntegerDeserializer();
      assertFalse(numberDeserializers_BigIntegerDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Class<Long> class0 = Long.class;
      Long long0 = new Long(0L);
      NumberDeserializers.LongDeserializer numberDeserializers_LongDeserializer0 = new NumberDeserializers.LongDeserializer(class0, long0);
      assertTrue(numberDeserializers_LongDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<Long> class0 = Long.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      NumberDeserializers.BigDecimalDeserializer numberDeserializers_BigDecimalDeserializer0 = new NumberDeserializers.BigDecimalDeserializer();
      assertFalse(numberDeserializers_BigDecimalDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      Integer integer0 = new Integer((-4341));
      NumberDeserializers.IntegerDeserializer numberDeserializers_IntegerDeserializer0 = new NumberDeserializers.IntegerDeserializer(class0, integer0);
      assertTrue(numberDeserializers_IntegerDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      NumberDeserializers numberDeserializers0 = new NumberDeserializers();
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<Character> class0 = Character.class;
      Character character0 = Character.valueOf(',');
      NumberDeserializers.CharacterDeserializer numberDeserializers_CharacterDeserializer0 = new NumberDeserializers.CharacterDeserializer(class0, character0);
      assertFalse(numberDeserializers_CharacterDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<Short> class0 = Short.class;
      Short short0 = new Short((short) (-2769));
      NumberDeserializers.ShortDeserializer numberDeserializers_ShortDeserializer0 = new NumberDeserializers.ShortDeserializer(class0, short0);
      assertFalse(numberDeserializers_ShortDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      NumberDeserializers.ShortDeserializer numberDeserializers_ShortDeserializer0 = NumberDeserializers.ShortDeserializer.primitiveInstance;
      JsonFactory jsonFactory0 = new JsonFactory();
      char[] charArray0 = new char[6];
      JsonParser jsonParser0 = jsonFactory0.createParser(charArray0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      try { 
        numberDeserializers_ShortDeserializer0.deserialize(jsonParser0, (DeserializationContext) defaultDeserializationContext_Impl0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not deserialize instance of short out of null token
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<Byte> class0 = Byte.TYPE;
      Byte byte0 = new Byte((byte) (-6));
      NumberDeserializers.ByteDeserializer numberDeserializers_ByteDeserializer0 = new NumberDeserializers.ByteDeserializer(class0, byte0);
      assertFalse(numberDeserializers_ByteDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      NumberDeserializers.ByteDeserializer numberDeserializers_ByteDeserializer0 = NumberDeserializers.ByteDeserializer.wrapperInstance;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        numberDeserializers_ByteDeserializer0.deserialize((JsonParser) null, (DeserializationContext) defaultDeserializationContext_Impl0);
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
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonFactory jsonFactory0 = new JsonFactory();
      NumberDeserializers.BooleanDeserializer numberDeserializers_BooleanDeserializer0 = NumberDeserializers.BooleanDeserializer.primitiveInstance;
      byte[] byteArray0 = new byte[1];
      JsonParser jsonParser0 = jsonFactory0.createParser(byteArray0, (-3831), (-3831));
      TokenFilter tokenFilter0 = TokenFilter.INCLUDE_ALL;
      FilteringParserDelegate filteringParserDelegate0 = new FilteringParserDelegate(jsonParser0, tokenFilter0, false, false);
      try { 
        numberDeserializers_BooleanDeserializer0.deserializeWithType(filteringParserDelegate0, defaultDeserializationContext_Impl0, (TypeDeserializer) null);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not deserialize instance of boolean out of null token
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Class<Boolean> class0 = Boolean.class;
      Boolean boolean0 = new Boolean("k z");
      NumberDeserializers.BooleanDeserializer numberDeserializers_BooleanDeserializer0 = new NumberDeserializers.BooleanDeserializer(class0, boolean0);
      assertFalse(numberDeserializers_BooleanDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      NumberDeserializers.BooleanDeserializer numberDeserializers_BooleanDeserializer0 = NumberDeserializers.BooleanDeserializer.primitiveInstance;
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper();
      Enumeration<SequenceInputStream> enumeration0 = (Enumeration<SequenceInputStream>) mock(Enumeration.class, new ViolatedAssumptionAnswer());
      doReturn(false).when(enumeration0).hasMoreElements();
      SequenceInputStream sequenceInputStream0 = new SequenceInputStream(enumeration0);
      PushbackInputStream pushbackInputStream0 = new PushbackInputStream(sequenceInputStream0);
      JsonParser jsonParser0 = jsonFactory0.createParser((InputStream) pushbackInputStream0);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      try { 
        numberDeserializers_BooleanDeserializer0.deserialize(jsonParser0, deserializationContext0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not deserialize instance of boolean out of null token
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Class<Float> class0 = Float.class;
      Float float0 = new Float(0.0F);
      NumberDeserializers.FloatDeserializer numberDeserializers_FloatDeserializer0 = new NumberDeserializers.FloatDeserializer(class0, float0);
      assertFalse(numberDeserializers_FloatDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      NumberDeserializers.FloatDeserializer numberDeserializers_FloatDeserializer0 = NumberDeserializers.FloatDeserializer.wrapperInstance;
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonParserSequence jsonParserSequence0 = JsonParserSequence.createFlattened((JsonParser) null, (JsonParser) null);
      // Undeclared exception!
      try { 
        numberDeserializers_FloatDeserializer0.deserialize((JsonParser) jsonParserSequence0, (DeserializationContext) defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.util.JsonParserDelegate", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonParser jsonParser0 = jsonFactory0.createParser("JSON");
      NumberDeserializers.DoubleDeserializer numberDeserializers_DoubleDeserializer0 = NumberDeserializers.DoubleDeserializer.primitiveInstance;
      try { 
        numberDeserializers_DoubleDeserializer0.deserialize(jsonParser0, (DeserializationContext) defaultDeserializationContext_Impl0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not deserialize instance of double out of null token
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0, defaultSerializerProvider_Impl0, defaultDeserializationContext_Impl0);
      Class<Double> class0 = Double.class;
      Double double0 = new Double(0.0);
      NumberDeserializers.DoubleDeserializer numberDeserializers_DoubleDeserializer0 = new NumberDeserializers.DoubleDeserializer(class0, double0);
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(true);
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      JsonParser jsonParser0 = arrayNode0.traverse((ObjectCodec) objectMapper0);
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
      JsonDeserializer<?> jsonDeserializer0 = NumberDeserializers.find(class0, "java.math.BigInteger");
      assertFalse(jsonDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Class<Integer> class0 = Integer.TYPE;
      JsonDeserializer<?> jsonDeserializer0 = NumberDeserializers.find(class0, "java.math.BigInteger");
      assertTrue(jsonDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Class<Boolean> class0 = Boolean.TYPE;
      JsonDeserializer<?> jsonDeserializer0 = NumberDeserializers.find(class0, (String) null);
      assertFalse(jsonDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Class<Long> class0 = Long.TYPE;
      JsonDeserializer<?> jsonDeserializer0 = NumberDeserializers.find(class0, "v-9)G>v");
      assertTrue(jsonDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Class<Double> class0 = Double.TYPE;
      JsonDeserializer<?> jsonDeserializer0 = NumberDeserializers.find(class0, "i:[v5cK|M^");
      assertFalse(jsonDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Class<Character> class0 = Character.TYPE;
      JsonDeserializer<?> jsonDeserializer0 = NumberDeserializers.find(class0, "java.math.BigInteger");
      assertFalse(jsonDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Class<Short> class0 = Short.TYPE;
      JsonDeserializer<?> jsonDeserializer0 = NumberDeserializers.find(class0, "java.math.BigInteger");
      assertFalse(jsonDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Class<Float> class0 = Float.TYPE;
      JsonDeserializer<?> jsonDeserializer0 = NumberDeserializers.find(class0, (String) null);
      assertFalse(jsonDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Class<Character> class0 = Character.class;
      JsonDeserializer<?> jsonDeserializer0 = NumberDeserializers.find(class0, "JSON");
      assertNull(jsonDeserializer0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Class<Boolean> class0 = Boolean.class;
      JsonDeserializer<?> jsonDeserializer0 = NumberDeserializers.find(class0, "java.math.BigInteger");
      assertNotNull(jsonDeserializer0);
      assertFalse(jsonDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Class<ChronoLocalDate> class0 = ChronoLocalDate.class;
      // Undeclared exception!
      try { 
        NumberDeserializers.find(class0, "java.math.BigInteger");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Internal error: can't find deserializer for java.time.chrono.ChronoLocalDate
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.NumberDeserializers", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Class<Double> class0 = Double.class;
      JsonDeserializer<?> jsonDeserializer0 = NumberDeserializers.find(class0, "java.math.BigInteger");
      assertFalse(jsonDeserializer0.isCachable());
      assertNotNull(jsonDeserializer0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Class<Character> class0 = Character.class;
      JsonDeserializer<?> jsonDeserializer0 = NumberDeserializers.find(class0, "java.math.BigInteger");
      assertNotNull(jsonDeserializer0);
      assertFalse(jsonDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Class<Byte> class0 = Byte.class;
      JsonDeserializer<?> jsonDeserializer0 = NumberDeserializers.find(class0, "java.math.BigInteger");
      assertNotNull(jsonDeserializer0);
      assertFalse(jsonDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Class<Short> class0 = Short.class;
      JsonDeserializer<?> jsonDeserializer0 = NumberDeserializers.find(class0, "java.math.BigInteger");
      assertFalse(jsonDeserializer0.isCachable());
      assertNotNull(jsonDeserializer0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Class<Float> class0 = Float.class;
      JsonDeserializer<?> jsonDeserializer0 = NumberDeserializers.find(class0, "java.math.BigInteger");
      assertNotNull(jsonDeserializer0);
      assertFalse(jsonDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0, (DefaultSerializerProvider) null, (DefaultDeserializationContext) null);
      Class<Object> class0 = Object.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Class<BigDecimal> class0 = BigDecimal.class;
      JsonDeserializer<?> jsonDeserializer0 = NumberDeserializers.find(class0, "java.math.BigInteger");
      assertNotNull(jsonDeserializer0);
      assertFalse(jsonDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Class<BigInteger> class0 = BigInteger.class;
      JsonDeserializer<?> jsonDeserializer0 = NumberDeserializers.find(class0, "java.math.BigInteger");
      assertFalse(jsonDeserializer0.isCachable());
      assertNotNull(jsonDeserializer0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      NumberDeserializers.DoubleDeserializer numberDeserializers_DoubleDeserializer0 = NumberDeserializers.DoubleDeserializer.wrapperInstance;
      Double double0 = numberDeserializers_DoubleDeserializer0.getNullValue((DeserializationContext) null);
      assertNull(double0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      NumberDeserializers.DoubleDeserializer numberDeserializers_DoubleDeserializer0 = NumberDeserializers.DoubleDeserializer.primitiveInstance;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Double double0 = numberDeserializers_DoubleDeserializer0.getNullValue((DeserializationContext) defaultDeserializationContext_Impl0);
      assertEquals(0.0, (double)double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonParser jsonParser0 = jsonFactory0.createParser("JSON");
      NumberDeserializers.CharacterDeserializer numberDeserializers_CharacterDeserializer0 = NumberDeserializers.CharacterDeserializer.wrapperInstance;
      try { 
        numberDeserializers_CharacterDeserializer0.deserialize(jsonParser0, (DeserializationContext) defaultDeserializationContext_Impl0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not deserialize instance of java.lang.Character out of null token
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonParser jsonParser0 = jsonFactory0.createParser("JSON");
      NumberDeserializers.IntegerDeserializer numberDeserializers_IntegerDeserializer0 = NumberDeserializers.IntegerDeserializer.wrapperInstance;
      try { 
        numberDeserializers_IntegerDeserializer0.deserialize(jsonParser0, (DeserializationContext) defaultDeserializationContext_Impl0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not deserialize instance of java.lang.Integer out of null token
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonParser jsonParser0 = jsonFactory0.createParser("JSON");
      NumberDeserializers.IntegerDeserializer numberDeserializers_IntegerDeserializer0 = NumberDeserializers.IntegerDeserializer.primitiveInstance;
      try { 
        numberDeserializers_IntegerDeserializer0.deserializeWithType(jsonParser0, defaultDeserializationContext_Impl0, (TypeDeserializer) null);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not deserialize instance of int out of null token
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonParser jsonParser0 = jsonFactory0.createParser("JSON");
      NumberDeserializers.LongDeserializer numberDeserializers_LongDeserializer0 = NumberDeserializers.LongDeserializer.primitiveInstance;
      try { 
        numberDeserializers_LongDeserializer0.deserialize(jsonParser0, (DeserializationContext) defaultDeserializationContext_Impl0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not deserialize instance of long out of null token
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonParser jsonParser0 = jsonFactory0.createParser("JSON");
      NumberDeserializers.NumberDeserializer numberDeserializers_NumberDeserializer0 = NumberDeserializers.NumberDeserializer.instance;
      try { 
        numberDeserializers_NumberDeserializer0.deserialize(jsonParser0, (DeserializationContext) defaultDeserializationContext_Impl0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not deserialize instance of java.lang.Number out of null token
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonParser jsonParser0 = jsonFactory0.createParser("JSON");
      NumberDeserializers.NumberDeserializer numberDeserializers_NumberDeserializer0 = new NumberDeserializers.NumberDeserializer();
      // Undeclared exception!
      try { 
        numberDeserializers_NumberDeserializer0.deserializeWithType(jsonParser0, defaultDeserializationContext_Impl0, (TypeDeserializer) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.NumberDeserializers$NumberDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonParser jsonParser0 = jsonFactory0.createParser("JSON");
      NumberDeserializers.BigIntegerDeserializer numberDeserializers_BigIntegerDeserializer0 = NumberDeserializers.BigIntegerDeserializer.instance;
      try { 
        numberDeserializers_BigIntegerDeserializer0.deserialize(jsonParser0, (DeserializationContext) defaultDeserializationContext_Impl0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not deserialize instance of java.math.BigInteger out of null token
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonParser jsonParser0 = jsonFactory0.createParser("JSON");
      NumberDeserializers.BigDecimalDeserializer numberDeserializers_BigDecimalDeserializer0 = NumberDeserializers.BigDecimalDeserializer.instance;
      try { 
        numberDeserializers_BigDecimalDeserializer0.deserialize(jsonParser0, (DeserializationContext) defaultDeserializationContext_Impl0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not deserialize instance of java.math.BigDecimal out of null token
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }
}
