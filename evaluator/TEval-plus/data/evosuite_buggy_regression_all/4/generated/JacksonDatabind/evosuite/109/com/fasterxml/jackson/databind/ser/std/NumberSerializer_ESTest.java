/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:48:24 GMT 2023
 */

package com.fasterxml.jackson.databind.ser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.util.BufferRecycler;
import com.fasterxml.jackson.core.util.ByteArrayBuilder;
import com.fasterxml.jackson.core.util.JsonGeneratorDelegate;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.jsonFormatVisitors.JsonFormatVisitorWrapper;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.ser.std.NumberSerializer;
import com.fasterxml.jackson.databind.type.PlaceholderForType;
import java.io.OutputStream;
import java.lang.reflect.Type;
import java.math.BigDecimal;
import java.math.BigInteger;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class NumberSerializer_ESTest extends NumberSerializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<BigInteger> class0 = BigInteger.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      PlaceholderForType placeholderForType0 = new PlaceholderForType(51);
      JsonFormatVisitorWrapper.Base jsonFormatVisitorWrapper_Base0 = new JsonFormatVisitorWrapper.Base();
      numberSerializer0.acceptJsonFormatVisitor(jsonFormatVisitorWrapper_Base0, placeholderForType0);
      assertFalse(placeholderForType0.isMapLikeType());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      NumberSerializer numberSerializer0 = NumberSerializer.instance;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Float float0 = new Float(2041.3);
      // Undeclared exception!
      try { 
        numberSerializer0.serialize((Number) float0, (JsonGenerator) null, (SerializerProvider) defaultSerializerProvider_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.NumberSerializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      NumberSerializer numberSerializer0 = NumberSerializer.instance;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      BigDecimal bigDecimal0 = new BigDecimal(1);
      // Undeclared exception!
      try { 
        numberSerializer0.instance.serialize((Number) bigDecimal0, (JsonGenerator) null, (SerializerProvider) defaultSerializerProvider_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.NumberSerializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      NumberSerializer numberSerializer0 = NumberSerializer.instance;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      byte[] byteArray0 = new byte[11];
      BigInteger bigInteger0 = new BigInteger(byteArray0);
      // Undeclared exception!
      try { 
        numberSerializer0.instance.serialize((Number) bigInteger0, (JsonGenerator) null, (SerializerProvider) defaultSerializerProvider_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.NumberSerializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      NumberSerializer numberSerializer0 = NumberSerializer.instance;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Long long0 = new Long(0L);
      // Undeclared exception!
      try { 
        numberSerializer0.serialize((Number) long0, (JsonGenerator) null, (SerializerProvider) defaultSerializerProvider_Impl0);
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
      NumberSerializer numberSerializer0 = NumberSerializer.instance;
      Double double0 = new Double(991.080632876);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
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
  public void test06()  throws Throwable  {
      NumberSerializer numberSerializer0 = NumberSerializer.instance;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        numberSerializer0.serialize((Number) null, (JsonGenerator) null, (SerializerProvider) defaultSerializerProvider_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.evosuite.runtime.System", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      NumberSerializer numberSerializer0 = NumberSerializer.instance;
      JsonFactory jsonFactory0 = new JsonFactory();
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      ByteArrayBuilder byteArrayBuilder0 = new ByteArrayBuilder(bufferRecycler0, 3);
      JsonGenerator jsonGenerator0 = jsonFactory0.createGenerator((OutputStream) byteArrayBuilder0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      numberSerializer0.serialize((Number) 1, jsonGenerator0, (SerializerProvider) defaultSerializerProvider_Impl0);
      assertEquals(1, jsonGenerator0.getOutputBuffered());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      NumberSerializer numberSerializer0 = NumberSerializer.instance;
      Byte byte0 = new Byte((byte) (-96));
      // Undeclared exception!
      try { 
        numberSerializer0.serialize((Number) byte0, (JsonGenerator) null, (SerializerProvider) null);
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
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Short short0 = new Short((short)1516);
      JsonGeneratorDelegate jsonGeneratorDelegate0 = new JsonGeneratorDelegate((JsonGenerator) null);
      // Undeclared exception!
      try { 
        numberSerializer0.serialize((Number) short0, (JsonGenerator) jsonGeneratorDelegate0, (SerializerProvider) defaultSerializerProvider_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.util.JsonGeneratorDelegate", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<Short> class0 = Short.class;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      JsonNode jsonNode0 = numberSerializer0.getSchema((SerializerProvider) defaultSerializerProvider_Impl0, (Type) class0);
      assertEquals(1, jsonNode0.size());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Class<BigInteger> class0 = BigInteger.class;
      Class<Object> class1 = Object.class;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      JsonNode jsonNode0 = numberSerializer0.getSchema((SerializerProvider) defaultSerializerProvider_Impl0, (Type) class1);
      assertEquals(1, jsonNode0.size());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      PlaceholderForType placeholderForType0 = new PlaceholderForType((-1795));
      JsonFormatVisitorWrapper.Base jsonFormatVisitorWrapper_Base0 = new JsonFormatVisitorWrapper.Base();
      Class<BigDecimal> class0 = BigDecimal.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      numberSerializer0.acceptJsonFormatVisitor(jsonFormatVisitorWrapper_Base0, placeholderForType0);
      assertFalse(numberSerializer0.usesObjectId());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Class<Short> class0 = Short.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      PlaceholderForType placeholderForType0 = new PlaceholderForType(51);
      JsonFormatVisitorWrapper.Base jsonFormatVisitorWrapper_Base0 = new JsonFormatVisitorWrapper.Base();
      numberSerializer0.acceptJsonFormatVisitor(jsonFormatVisitorWrapper_Base0, placeholderForType0);
      assertFalse(placeholderForType0.isPrimitive());
  }
}
