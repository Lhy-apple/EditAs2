/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:22:07 GMT 2023
 */

package com.fasterxml.jackson.databind.ser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.Base64Variant;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.base.GeneratorBase;
import com.fasterxml.jackson.core.io.IOContext;
import com.fasterxml.jackson.core.json.ReaderBasedJsonParser;
import com.fasterxml.jackson.core.json.UTF8JsonGenerator;
import com.fasterxml.jackson.core.json.WriterBasedJsonGenerator;
import com.fasterxml.jackson.core.sym.CharsToNameCanonicalizer;
import com.fasterxml.jackson.core.util.BufferRecycler;
import com.fasterxml.jackson.core.util.ByteArrayBuilder;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.jsonFormatVisitors.JsonFormatVisitorWrapper;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.ser.std.NumberSerializer;
import com.fasterxml.jackson.databind.type.SimpleType;
import java.io.CharArrayWriter;
import java.io.Reader;
import java.io.Writer;
import java.lang.reflect.Type;
import java.math.BigDecimal;
import java.math.BigInteger;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockPrintWriter;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class NumberSerializer_ESTest extends NumberSerializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<BigInteger> class0 = BigInteger.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Double double0 = new Double(0.0);
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, (Object) null, true);
      ObjectMapper objectMapper0 = new ObjectMapper((JsonFactory) null);
      ObjectReader objectReader0 = objectMapper0.reader((Base64Variant) null);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      char[] charArray0 = new char[1];
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 2, (Reader) null, objectReader0, charsToNameCanonicalizer0, charArray0, 0, 1990, true);
      ByteArrayBuilder byteArrayBuilder0 = readerBasedJsonParser0._getByteArrayBuilder();
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 0, objectReader0, byteArrayBuilder0, byteArrayBuilder0.NO_BYTES, 3, true);
      // Undeclared exception!
      try { 
        numberSerializer0.serialize((Number) double0, (JsonGenerator) uTF8JsonGenerator0, (SerializerProvider) defaultSerializerProvider_Impl0);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      NumberSerializer numberSerializer0 = NumberSerializer.instance;
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      Object object0 = new Object();
      IOContext iOContext0 = new IOContext(bufferRecycler0, object0, false);
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      ByteArrayBuilder byteArrayBuilder0 = new ByteArrayBuilder(bufferRecycler0, 2);
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, (-700), objectMapper0, byteArrayBuilder0);
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProvider();
      numberSerializer0.serialize((Number) 55296, (JsonGenerator) uTF8JsonGenerator0, serializerProvider0);
      assertEquals(5, uTF8JsonGenerator0.getOutputBuffered());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      NumberSerializer numberSerializer0 = NumberSerializer.instance;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      BigDecimal bigDecimal0 = BigDecimal.ONE;
      // Undeclared exception!
      try { 
        numberSerializer0.serialize((Number) bigDecimal0, (JsonGenerator) null, (SerializerProvider) defaultSerializerProvider_Impl0);
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
      Class<Float> class0 = Float.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      BigInteger bigInteger0 = BigInteger.TEN;
      JsonFactory jsonFactory0 = new JsonFactory();
      CharArrayWriter charArrayWriter0 = new CharArrayWriter();
      MockPrintWriter mockPrintWriter0 = new MockPrintWriter(charArrayWriter0);
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = (WriterBasedJsonGenerator)jsonFactory0.createGenerator((Writer) mockPrintWriter0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      numberSerializer0.serialize((Number) bigInteger0, (JsonGenerator) writerBasedJsonGenerator0, (SerializerProvider) defaultSerializerProvider_Impl0);
      assertEquals(55296, GeneratorBase.SURR1_FIRST);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
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
  public void test05()  throws Throwable  {
      NumberSerializer numberSerializer0 = NumberSerializer.instance;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Long long0 = new Long(2691L);
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
  public void test06()  throws Throwable  {
      NumberSerializer numberSerializer0 = NumberSerializer.instance;
      Float float0 = new Float((-1238.21908925432));
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
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
  public void test07()  throws Throwable  {
      NumberSerializer numberSerializer0 = NumberSerializer.instance;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Byte byte0 = new Byte((byte)0);
      // Undeclared exception!
      try { 
        numberSerializer0.serialize((Number) byte0, (JsonGenerator) null, (SerializerProvider) defaultSerializerProvider_Impl0);
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
      Short short0 = new Short((short)0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
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
  public void test09()  throws Throwable  {
      NumberSerializer numberSerializer0 = NumberSerializer.instance;
      ObjectMapper objectMapper0 = new ObjectMapper();
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProviderInstance();
      JsonNode jsonNode0 = numberSerializer0.getSchema(serializerProvider0, (Type) null);
      assertEquals(1, jsonNode0.size());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<BigInteger> class0 = BigInteger.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonNode jsonNode0 = numberSerializer0.getSchema((SerializerProvider) defaultSerializerProvider_Impl0, (Type) class0);
      assertEquals(1, jsonNode0.size());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      NumberSerializer numberSerializer0 = NumberSerializer.instance;
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProvider();
      JsonFormatVisitorWrapper.Base jsonFormatVisitorWrapper_Base0 = new JsonFormatVisitorWrapper.Base(serializerProvider0);
      Class<BigInteger> class0 = BigInteger.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      numberSerializer0.acceptJsonFormatVisitor(jsonFormatVisitorWrapper_Base0, simpleType0);
      assertFalse(simpleType0.isEnumType());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<BigInteger> class0 = BigInteger.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      JsonFormatVisitorWrapper.Base jsonFormatVisitorWrapper_Base0 = new JsonFormatVisitorWrapper.Base();
      numberSerializer0.acceptJsonFormatVisitor(jsonFormatVisitorWrapper_Base0, (JavaType) null);
      assertFalse(numberSerializer0.usesObjectId());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProvider();
      JsonFormatVisitorWrapper.Base jsonFormatVisitorWrapper_Base0 = new JsonFormatVisitorWrapper.Base(serializerProvider0);
      Class<BigDecimal> class0 = BigDecimal.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      Class<BigInteger> class1 = BigInteger.class;
      SimpleType simpleType0 = SimpleType.construct(class1);
      numberSerializer0.acceptJsonFormatVisitor(jsonFormatVisitorWrapper_Base0, simpleType0);
      assertTrue(simpleType0.isConcrete());
  }
}
