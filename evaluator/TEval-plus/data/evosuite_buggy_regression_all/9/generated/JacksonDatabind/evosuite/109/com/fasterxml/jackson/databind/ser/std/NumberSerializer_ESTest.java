/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:59:22 GMT 2023
 */

package com.fasterxml.jackson.databind.ser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.io.IOContext;
import com.fasterxml.jackson.core.json.UTF8JsonGenerator;
import com.fasterxml.jackson.core.util.BufferRecycler;
import com.fasterxml.jackson.core.util.ByteArrayBuilder;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.jsonFormatVisitors.JsonFormatVisitorWrapper;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.ser.std.NumberSerializer;
import com.fasterxml.jackson.databind.type.PlaceholderForType;
import java.lang.reflect.Type;
import java.math.BigDecimal;
import java.math.BigInteger;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class NumberSerializer_ESTest extends NumberSerializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<BigDecimal> class0 = BigDecimal.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonNode jsonNode0 = numberSerializer0.getSchema((SerializerProvider) defaultSerializerProvider_Impl0, (Type) class0);
      assertEquals(1, jsonNode0.size());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Class<BigInteger> class0 = BigInteger.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonNode jsonNode0 = numberSerializer0.getSchema((SerializerProvider) defaultSerializerProvider_Impl0, (Type) class0);
      assertEquals(1, jsonNode0.size());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      NumberSerializer numberSerializer0 = NumberSerializer.instance;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      BeanProperty.Bogus beanProperty_Bogus0 = new BeanProperty.Bogus();
      JsonSerializer<?> jsonSerializer0 = numberSerializer0.createContextual(defaultSerializerProvider_Impl0, beanProperty_Bogus0);
      assertFalse(jsonSerializer0.isUnwrappingSerializer());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      NumberSerializer numberSerializer0 = NumberSerializer.instance;
      ObjectMapper objectMapper0 = new ObjectMapper();
      Byte byte0 = new Byte((byte) (-128));
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, objectMapper0, false);
      ByteArrayBuilder byteArrayBuilder0 = new ByteArrayBuilder();
      MockPrintStream mockPrintStream0 = new MockPrintStream(byteArrayBuilder0);
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 0, objectMapper0, mockPrintStream0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      numberSerializer0.serialize((Number) byte0, (JsonGenerator) uTF8JsonGenerator0, (SerializerProvider) defaultSerializerProvider_Impl0);
      assertEquals(4, uTF8JsonGenerator0.getOutputBuffered());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      NumberSerializer numberSerializer0 = NumberSerializer.instance;
      ObjectMapper objectMapper0 = new ObjectMapper();
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProviderInstance();
      BigDecimal bigDecimal0 = BigDecimal.ZERO;
      // Undeclared exception!
      try { 
        numberSerializer0.serialize((Number) bigDecimal0, (JsonGenerator) null, serializerProvider0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.NumberSerializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<Double> class0 = Double.TYPE;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      BigInteger bigInteger0 = BigInteger.TEN;
      // Undeclared exception!
      try { 
        numberSerializer0.serialize((Number) bigInteger0, (JsonGenerator) null, (SerializerProvider) defaultSerializerProvider_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.NumberSerializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      NumberSerializer numberSerializer0 = NumberSerializer.instance;
      ObjectMapper objectMapper0 = new ObjectMapper();
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProviderInstance();
      Long long0 = new Long(0L);
      // Undeclared exception!
      try { 
        numberSerializer0.serialize((Number) long0, (JsonGenerator) null, serializerProvider0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.NumberSerializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      NumberSerializer numberSerializer0 = NumberSerializer.instance;
      Double double0 = new Double((byte) (-103));
      // Undeclared exception!
      try { 
        numberSerializer0.serialize((Number) double0, (JsonGenerator) null, (SerializerProvider) defaultSerializerProvider_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.NumberSerializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      NumberSerializer numberSerializer0 = NumberSerializer.instance;
      ObjectMapper objectMapper0 = new ObjectMapper();
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProviderInstance();
      Float float0 = Float.valueOf(0.0F);
      // Undeclared exception!
      try { 
        numberSerializer0.serialize((Number) float0, (JsonGenerator) null, serializerProvider0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.NumberSerializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      NumberSerializer numberSerializer0 = NumberSerializer.instance;
      ObjectMapper objectMapper0 = new ObjectMapper();
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProvider();
      // Undeclared exception!
      try { 
        numberSerializer0.serialize((Number) 3, (JsonGenerator) null, serializerProvider0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.NumberSerializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<Double> class0 = Double.TYPE;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Short short0 = new Short((short)3492);
      // Undeclared exception!
      try { 
        numberSerializer0.serialize((Number) short0, (JsonGenerator) null, (SerializerProvider) defaultSerializerProvider_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.NumberSerializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      NumberSerializer numberSerializer0 = NumberSerializer.instance;
      ObjectMapper objectMapper0 = new ObjectMapper();
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProviderInstance();
      // Undeclared exception!
      try { 
        numberSerializer0.serialize((Number) null, (JsonGenerator) null, serializerProvider0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.evosuite.runtime.System", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<BigDecimal> class0 = BigDecimal.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      JsonFormatVisitorWrapper.Base jsonFormatVisitorWrapper_Base0 = new JsonFormatVisitorWrapper.Base();
      PlaceholderForType placeholderForType0 = new PlaceholderForType(4962);
      numberSerializer0.acceptJsonFormatVisitor(jsonFormatVisitorWrapper_Base0, placeholderForType0);
      assertFalse(placeholderForType0.isAbstract());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Class<BigInteger> class0 = BigInteger.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      JsonFormatVisitorWrapper.Base jsonFormatVisitorWrapper_Base0 = new JsonFormatVisitorWrapper.Base();
      PlaceholderForType placeholderForType0 = new PlaceholderForType(968);
      numberSerializer0.acceptJsonFormatVisitor(jsonFormatVisitorWrapper_Base0, placeholderForType0);
      assertFalse(placeholderForType0.isPrimitive());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      NumberSerializer numberSerializer0 = NumberSerializer.instance;
      JsonFormatVisitorWrapper.Base jsonFormatVisitorWrapper_Base0 = new JsonFormatVisitorWrapper.Base();
      numberSerializer0.acceptJsonFormatVisitor(jsonFormatVisitorWrapper_Base0, (JavaType) null);
      assertFalse(numberSerializer0.usesObjectId());
  }
}
