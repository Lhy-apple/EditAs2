/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:39:28 GMT 2023
 */

package com.fasterxml.jackson.databind.ser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.ObjectIdGenerator;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.jsonFormatVisitors.JsonFormatVisitorWrapper;
import com.fasterxml.jackson.databind.jsontype.impl.AsWrapperTypeSerializer;
import com.fasterxml.jackson.databind.ser.BeanPropertyWriter;
import com.fasterxml.jackson.databind.ser.BeanSerializer;
import com.fasterxml.jackson.databind.ser.BeanSerializerBuilder;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.ser.impl.IndexedListSerializer;
import com.fasterxml.jackson.databind.ser.impl.ObjectIdWriter;
import com.fasterxml.jackson.databind.ser.std.BeanSerializerBase;
import com.fasterxml.jackson.databind.ser.std.CollectionSerializer;
import com.fasterxml.jackson.databind.ser.std.MapSerializer;
import com.fasterxml.jackson.databind.ser.std.StdArraySerializers;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import com.fasterxml.jackson.databind.util.NameTransformer;
import java.lang.reflect.Type;
import java.util.HashMap;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BeanSerializerBase_ESTest extends BeanSerializerBase_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      StdArraySerializers.LongArraySerializer stdArraySerializers_LongArraySerializer0 = new StdArraySerializers.LongArraySerializer();
      JavaType javaType0 = stdArraySerializers_LongArraySerializer0.getContentType();
      BeanSerializer beanSerializer0 = BeanSerializer.createDummy(javaType0);
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      ObjectIdGenerator<IndexedListSerializer> objectIdGenerator0 = (ObjectIdGenerator<IndexedListSerializer>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      ObjectIdWriter objectIdWriter0 = ObjectIdWriter.construct(javaType0, propertyName0, objectIdGenerator0, false);
      BeanSerializerBase beanSerializerBase0 = beanSerializer0.withObjectIdWriter(objectIdWriter0);
      boolean boolean0 = beanSerializerBase0.usesObjectId();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)MapSerializer.UNSPECIFIED_TYPE;
      BeanSerializer beanSerializer0 = BeanSerializer.createDummy(simpleType0);
      NameTransformer nameTransformer0 = NameTransformer.NOP;
      JsonSerializer<Object> jsonSerializer0 = beanSerializer0.unwrappingSerializer(nameTransformer0);
      assertFalse(jsonSerializer0.usesObjectId());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      // Undeclared exception!
      try { 
        objectMapper0.writeValueAsBytes(objectMapper0);
        fail("Expecting exception: NoClassDefFoundError");
      
      } catch(NoClassDefFoundError e) {
         //
         // Could not initialize class com.fasterxml.jackson.databind.JsonMappingException
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.BeanSerializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<ObjectIdGenerators.UUIDGenerator> class0 = ObjectIdGenerators.UUIDGenerator.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      BeanPropertyWriter[] beanPropertyWriterArray0 = new BeanPropertyWriter[0];
      BeanSerializer beanSerializer0 = new BeanSerializer(simpleType0, (BeanSerializerBuilder) null, beanPropertyWriterArray0, beanPropertyWriterArray0);
      beanSerializer0.resolve(defaultSerializerProvider_Impl0);
      assertFalse(beanSerializer0.usesObjectId());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      JsonFormatVisitorWrapper.Base jsonFormatVisitorWrapper_Base0 = new JsonFormatVisitorWrapper.Base();
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonInclude.Include jsonInclude_Include0 = JsonInclude.Include.NON_EMPTY;
      ObjectMapper objectMapper1 = objectMapper0.setSerializationInclusion(jsonInclude_Include0);
      Class<CollectionSerializer> class0 = CollectionSerializer.class;
      objectMapper1.acceptJsonFormatVisitor((Class<?>) class0, (JsonFormatVisitorWrapper) jsonFormatVisitorWrapper_Base0);
      assertSame(objectMapper1, objectMapper0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Class<StdArraySerializers.IntArraySerializer> class0 = StdArraySerializers.IntArraySerializer.class;
      SimpleType simpleType0 = (SimpleType)MapSerializer.UNSPECIFIED_TYPE;
      CollectionType collectionType0 = CollectionType.construct(class0, simpleType0);
      MapType mapType0 = MapType.construct(class0, collectionType0, collectionType0);
      BeanSerializer beanSerializer0 = BeanSerializer.createDummy(mapType0);
      boolean boolean0 = beanSerializer0.usesObjectId();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      ObjectIdGenerators.UUIDGenerator objectIdGenerators_UUIDGenerator0 = new ObjectIdGenerators.UUIDGenerator();
      ObjectMapper objectMapper0 = new ObjectMapper();
      byte[] byteArray0 = objectMapper0.writeValueAsBytes(objectIdGenerators_UUIDGenerator0);
      assertEquals(28, byteArray0.length);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)MapSerializer.UNSPECIFIED_TYPE;
      BeanSerializer beanSerializer0 = BeanSerializer.createDummy(simpleType0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<AsWrapperTypeSerializer> class0 = AsWrapperTypeSerializer.class;
      JsonNode jsonNode0 = beanSerializer0.getSchema((SerializerProvider) defaultSerializerProvider_Impl0, (Type) class0);
      assertEquals(2, jsonNode0.size());
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      BeanSerializer beanSerializer0 = BeanSerializer.createDummy(mapType0);
      beanSerializer0.acceptJsonFormatVisitor((JsonFormatVisitorWrapper) null, mapType0);
      assertFalse(beanSerializer0.isUnwrappingSerializer());
  }
}
