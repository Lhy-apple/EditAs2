/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:00:32 GMT 2023
 */

package com.fasterxml.jackson.databind.ser.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.ObjectIdGenerator;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.core.base.GeneratorBase;
import com.fasterxml.jackson.core.io.IOContext;
import com.fasterxml.jackson.core.io.SerializedString;
import com.fasterxml.jackson.core.json.UTF8JsonGenerator;
import com.fasterxml.jackson.core.util.BufferRecycler;
import com.fasterxml.jackson.core.util.ByteArrayBuilder;
import com.fasterxml.jackson.core.util.DefaultPrettyPrinter;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.ser.impl.ObjectIdWriter;
import com.fasterxml.jackson.databind.ser.impl.WritableObjectId;
import com.fasterxml.jackson.databind.type.CollectionLikeType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.SimpleType;
import java.io.IOException;
import java.io.OutputStream;
import java.time.chrono.ChronoLocalDate;
import java.time.temporal.ChronoField;
import java.time.temporal.TemporalField;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class WritableObjectId_ESTest extends WritableObjectId_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      ByteArrayBuilder byteArrayBuilder0 = new ByteArrayBuilder();
      JsonGenerator jsonGenerator0 = jsonFactory0.createGenerator((OutputStream) byteArrayBuilder0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      ObjectIdGenerators.IntSequenceGenerator objectIdGenerators_IntSequenceGenerator0 = new ObjectIdGenerators.IntSequenceGenerator();
      WritableObjectId writableObjectId0 = new WritableObjectId(objectIdGenerators_IntSequenceGenerator0);
      boolean boolean0 = writableObjectId0.writeAsId(jsonGenerator0, defaultSerializerProvider_Impl0, (ObjectIdWriter) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      ObjectIdGenerators.IntSequenceGenerator objectIdGenerators_IntSequenceGenerator0 = new ObjectIdGenerators.IntSequenceGenerator();
      WritableObjectId writableObjectId0 = new WritableObjectId(objectIdGenerators_IntSequenceGenerator0);
      writableObjectId0.generateId(objectIdGenerators_IntSequenceGenerator0);
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      writableObjectId0.idWritten = true;
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 1, (ObjectCodec) null, (OutputStream) null);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        writableObjectId0.writeAsId(uTF8JsonGenerator0, defaultSerializerProvider_Impl0, (ObjectIdWriter) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.impl.WritableObjectId", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      ObjectIdGenerators.IntSequenceGenerator objectIdGenerators_IntSequenceGenerator0 = new ObjectIdGenerators.IntSequenceGenerator();
      Object object0 = new Object();
      WritableObjectId writableObjectId0 = new WritableObjectId(objectIdGenerators_IntSequenceGenerator0);
      Object object1 = writableObjectId0.generateId(object0);
      assertNotNull(object1);
      
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 3, (ObjectCodec) null, (OutputStream) null);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<ChronoField> class0 = ChronoField.class;
      Class<Integer> class1 = Integer.class;
      Class<ChronoLocalDate> class2 = ChronoLocalDate.class;
      SimpleType simpleType0 = SimpleType.construct(class2);
      CollectionLikeType collectionLikeType0 = CollectionLikeType.construct(class1, simpleType0);
      MapType mapType0 = MapType.construct(class0, collectionLikeType0, collectionLikeType0);
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      ObjectIdWriter objectIdWriter0 = ObjectIdWriter.construct((JavaType) mapType0, propertyName0, (ObjectIdGenerator<?>) objectIdGenerators_IntSequenceGenerator0, false);
      boolean boolean0 = writableObjectId0.writeAsId(uTF8JsonGenerator0, defaultSerializerProvider_Impl0, objectIdWriter0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      ObjectIdGenerators.IntSequenceGenerator objectIdGenerators_IntSequenceGenerator0 = new ObjectIdGenerators.IntSequenceGenerator();
      Object object0 = new Object();
      WritableObjectId writableObjectId0 = new WritableObjectId(objectIdGenerators_IntSequenceGenerator0);
      writableObjectId0.generateId(object0);
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 3, (ObjectCodec) null, (OutputStream) null);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<ChronoField> class0 = ChronoField.class;
      Class<Integer> class1 = Integer.class;
      Class<ChronoLocalDate> class2 = ChronoLocalDate.class;
      SimpleType simpleType0 = SimpleType.construct(class2);
      CollectionLikeType collectionLikeType0 = CollectionLikeType.construct(class1, simpleType0);
      MapType mapType0 = MapType.construct(class0, collectionLikeType0, collectionLikeType0);
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      ObjectIdWriter objectIdWriter0 = ObjectIdWriter.construct((JavaType) mapType0, propertyName0, (ObjectIdGenerator<?>) objectIdGenerators_IntSequenceGenerator0, true);
      // Undeclared exception!
      try { 
        writableObjectId0.writeAsId(uTF8JsonGenerator0, defaultSerializerProvider_Impl0, objectIdWriter0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.impl.WritableObjectId", e);
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      ByteArrayBuilder byteArrayBuilder0 = new ByteArrayBuilder();
      UTF8JsonGenerator uTF8JsonGenerator0 = (UTF8JsonGenerator)jsonFactory0.createGenerator((OutputStream) byteArrayBuilder0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      ObjectIdGenerators.IntSequenceGenerator objectIdGenerators_IntSequenceGenerator0 = new ObjectIdGenerators.IntSequenceGenerator();
      WritableObjectId writableObjectId0 = new WritableObjectId(objectIdGenerators_IntSequenceGenerator0);
      ObjectIdWriter objectIdWriter0 = ObjectIdWriter.construct((JavaType) null, (PropertyName) null, writableObjectId0.generator, true);
      writableObjectId0.writeAsField(uTF8JsonGenerator0, defaultSerializerProvider_Impl0, objectIdWriter0);
      assertEquals(56320, GeneratorBase.SURR2_FIRST);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      ByteArrayBuilder byteArrayBuilder0 = new ByteArrayBuilder();
      JsonGenerator jsonGenerator0 = jsonFactory0.createGenerator((OutputStream) byteArrayBuilder0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      ObjectIdGenerators.IntSequenceGenerator objectIdGenerators_IntSequenceGenerator0 = new ObjectIdGenerators.IntSequenceGenerator();
      WritableObjectId writableObjectId0 = new WritableObjectId(objectIdGenerators_IntSequenceGenerator0);
      Class<TemporalField> class0 = TemporalField.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      SerializedString serializedString0 = DefaultPrettyPrinter.DEFAULT_ROOT_VALUE_SEPARATOR;
      ObjectIdWriter objectIdWriter0 = new ObjectIdWriter(simpleType0, serializedString0, objectIdGenerators_IntSequenceGenerator0, defaultSerializerProvider_Impl0.DEFAULT_NULL_KEY_SERIALIZER, false);
      try { 
        writableObjectId0.writeAsField(jsonGenerator0, defaultSerializerProvider_Impl0, objectIdWriter0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Null key for a Map not allowed in JSON (use a converting NullKeySerializer?)
         //
         verifyException("com.fasterxml.jackson.databind.ser.impl.FailingSerializer", e);
      }
  }
}