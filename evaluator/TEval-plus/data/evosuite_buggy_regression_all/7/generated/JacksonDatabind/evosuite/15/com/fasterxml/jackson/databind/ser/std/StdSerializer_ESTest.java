/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:54:26 GMT 2023
 */

package com.fasterxml.jackson.databind.ser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.ObjectIdGenerator;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.core.io.IOContext;
import com.fasterxml.jackson.core.json.UTF8JsonGenerator;
import com.fasterxml.jackson.core.util.BufferRecycler;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.ObjectWriter;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.ser.std.BooleanSerializer;
import com.fasterxml.jackson.databind.ser.std.NumberSerializer;
import com.fasterxml.jackson.databind.ser.std.RawSerializer;
import com.fasterxml.jackson.databind.ser.std.StdArraySerializers;
import com.fasterxml.jackson.databind.ser.std.StringSerializer;
import java.io.IOException;
import java.io.PipedOutputStream;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Type;
import java.sql.BatchUpdateException;
import java.sql.SQLFeatureNotSupportedException;
import java.sql.SQLTransientException;
import java.time.chrono.ChronoLocalDate;
import java.util.concurrent.atomic.AtomicInteger;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockIOException;
import org.evosuite.runtime.mock.java.lang.MockError;
import org.evosuite.runtime.mock.java.lang.MockRuntimeException;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class StdSerializer_ESTest extends StdSerializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      AtomicInteger atomicInteger0 = new AtomicInteger(3833);
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, atomicInteger0, true);
      ObjectMapper objectMapper0 = new ObjectMapper();
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 2, objectMapper0, pipedOutputStream0);
      int[] intArray0 = new int[5];
      BatchUpdateException batchUpdateException0 = new BatchUpdateException("", intArray0);
      SQLFeatureNotSupportedException sQLFeatureNotSupportedException0 = new SQLFeatureNotSupportedException("", "", 2, batchUpdateException0);
      InvocationTargetException invocationTargetException0 = new InvocationTargetException(sQLFeatureNotSupportedException0, "(.58zkb");
      try { 
        objectMapper0.writeValue((JsonGenerator) uTF8JsonGenerator0, (Object) invocationTargetException0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Pipe not connected
         //
         verifyException("java.io.PipedOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Class<Object> class0 = Object.class;
      RawSerializer<IOException> rawSerializer0 = new RawSerializer<IOException>(class0);
      Class<IOException> class1 = rawSerializer0.handledType();
      assertFalse(class1.isEnum());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      RawSerializer<ObjectReader> rawSerializer0 = new RawSerializer<ObjectReader>(class0);
      boolean boolean0 = numberSerializer0.isDefaultSerializer(rawSerializer0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<Integer> class0 = Integer.TYPE;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      JsonNode jsonNode0 = numberSerializer0.getSchema((SerializerProvider) defaultSerializerProvider_Impl0, (Type) class0, true);
      assertEquals(1, jsonNode0.size());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonNode jsonNode0 = numberSerializer0.getSchema((SerializerProvider) defaultSerializerProvider_Impl0, (Type) class0, false);
      assertEquals(2, jsonNode0.size());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      BooleanSerializer booleanSerializer0 = new BooleanSerializer(true);
      Class<ObjectWriter> class0 = ObjectWriter.class;
      MockRuntimeException mockRuntimeException0 = new MockRuntimeException((Throwable) null);
      ObjectIdGenerator.IdKey objectIdGenerator_IdKey0 = new ObjectIdGenerator.IdKey(class0, class0, mockRuntimeException0);
      JsonNode jsonNode0 = booleanSerializer0.getSchema((SerializerProvider) null, (Type) objectIdGenerator_IdKey0.scope);
      assertEquals(2, jsonNode0.size());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Class<ChronoLocalDate> class0 = ChronoLocalDate.class;
      RawSerializer<Error> rawSerializer0 = new RawSerializer<Error>(class0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      MockError mockError0 = new MockError();
      // Undeclared exception!
      try { 
        rawSerializer0.wrapAndThrow((SerializerProvider) defaultSerializerProvider_Impl0, (Throwable) mockError0, (Object) mockError0, "");
        fail("Expecting exception: Error");
      
      } catch(Error e) {
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      SQLTransientException sQLTransientException0 = new SQLTransientException("t4STdR8^IG&ML#Pq#", "t4STdR8^IG&ML#Pq#");
      Class<ChronoLocalDate> class0 = ChronoLocalDate.class;
      RawSerializer<Error> rawSerializer0 = new RawSerializer<Error>(class0);
      try { 
        rawSerializer0.wrapAndThrow((SerializerProvider) null, (Throwable) sQLTransientException0, (Object) class0, "t4STdR8^IG&ML#Pq#");
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // t4STdR8^IG&ML#Pq# (through reference chain: java.time.chrono.ChronoLocalDate[\"t4STdR8^IG&ML#Pq#\"])
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<Error> class0 = Error.class;
      JsonToken jsonToken0 = JsonToken.VALUE_NULL;
      JsonMappingException jsonMappingException0 = defaultDeserializationContext_Impl0.mappingException(class0, jsonToken0);
      RawSerializer<Error> rawSerializer0 = new RawSerializer<Error>(class0);
      try { 
        rawSerializer0.wrapAndThrow((SerializerProvider) null, (Throwable) jsonMappingException0, (Object) defaultDeserializationContext_Impl0, "alrGlzJYV]RfCX");
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not deserialize instance of java.lang.Error out of VALUE_NULL token (through reference chain: com.fasterxml.jackson.databind.deser.Impl[\"alrGlzJYV]RfCX\"])
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      StdArraySerializers.DoubleArraySerializer stdArraySerializers_DoubleArraySerializer0 = new StdArraySerializers.DoubleArraySerializer();
      // Undeclared exception!
      try { 
        stdArraySerializers_DoubleArraySerializer0.wrapAndThrow((SerializerProvider) null, (Throwable) null, (Object) null, 1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<Object> class0 = Object.class;
      RawSerializer<InvocationTargetException> rawSerializer0 = new RawSerializer<InvocationTargetException>(class0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      StdArraySerializers.DoubleArraySerializer stdArraySerializers_DoubleArraySerializer0 = new StdArraySerializers.DoubleArraySerializer();
      InvocationTargetException invocationTargetException0 = new InvocationTargetException((Throwable) null);
      // Undeclared exception!
      try { 
        stdArraySerializers_DoubleArraySerializer0.wrapAndThrow((SerializerProvider) defaultSerializerProvider_Impl0, (Throwable) invocationTargetException0, (Object) rawSerializer0, 698);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      int[] intArray0 = new int[5];
      BatchUpdateException batchUpdateException0 = new BatchUpdateException("", intArray0);
      InvocationTargetException invocationTargetException0 = new InvocationTargetException(batchUpdateException0, "");
      StdArraySerializers.DoubleArraySerializer stdArraySerializers_DoubleArraySerializer0 = new StdArraySerializers.DoubleArraySerializer();
      // Undeclared exception!
      try { 
        stdArraySerializers_DoubleArraySerializer0.wrapAndThrow((SerializerProvider) defaultSerializerProvider_Impl0, (Throwable) invocationTargetException0, (Object) batchUpdateException0, 1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      StdArraySerializers.DoubleArraySerializer stdArraySerializers_DoubleArraySerializer0 = new StdArraySerializers.DoubleArraySerializer();
      MockError mockError0 = new MockError("");
      // Undeclared exception!
      try { 
        stdArraySerializers_DoubleArraySerializer0.wrapAndThrow((SerializerProvider) null, (Throwable) mockError0, (Object) null, (-2580));
        fail("Expecting exception: Error");
      
      } catch(Error e) {
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<Error> class0 = Error.class;
      JsonToken jsonToken0 = JsonToken.VALUE_NULL;
      JsonMappingException jsonMappingException0 = defaultDeserializationContext_Impl0.mappingException(class0, jsonToken0);
      StdArraySerializers.DoubleArraySerializer stdArraySerializers_DoubleArraySerializer0 = new StdArraySerializers.DoubleArraySerializer();
      try { 
        stdArraySerializers_DoubleArraySerializer0.wrapAndThrow((SerializerProvider) null, (Throwable) jsonMappingException0, (Object) class0, 1);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not deserialize instance of java.lang.Error out of VALUE_NULL token (through reference chain: java.lang.Error[1])
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      StringSerializer stringSerializer0 = new StringSerializer();
      MockIOException mockIOException0 = new MockIOException("");
      try { 
        stringSerializer0.wrapAndThrow((SerializerProvider) null, (Throwable) mockIOException0, (Object) "", 0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
      }
  }
}
