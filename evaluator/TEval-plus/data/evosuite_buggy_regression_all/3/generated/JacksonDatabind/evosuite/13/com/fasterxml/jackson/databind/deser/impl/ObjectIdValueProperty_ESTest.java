/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:37:13 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.ObjectIdGenerator;
import com.fasterxml.jackson.annotation.ObjectIdResolver;
import com.fasterxml.jackson.annotation.SimpleObjectIdResolver;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.core.io.IOContext;
import com.fasterxml.jackson.core.json.ReaderBasedJsonParser;
import com.fasterxml.jackson.core.sym.CharsToNameCanonicalizer;
import com.fasterxml.jackson.core.util.BufferRecycler;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.PropertyMetadata;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.SettableBeanProperty;
import com.fasterxml.jackson.databind.deser.impl.ObjectIdReader;
import com.fasterxml.jackson.databind.deser.impl.ObjectIdValueProperty;
import com.fasterxml.jackson.databind.introspect.AnnotatedMember;
import com.fasterxml.jackson.databind.node.POJONode;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.io.PipedReader;
import java.io.PipedWriter;
import java.lang.annotation.Annotation;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ObjectIdValueProperty_ESTest extends ObjectIdValueProperty_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      PropertyName propertyName0 = PropertyName.construct("g5vj3D`7", "g5vj3D`7");
      ObjectIdGenerator<Object> objectIdGenerator0 = (ObjectIdGenerator<Object>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      JsonDeserializer<ObjectReader> jsonDeserializer0 = (JsonDeserializer<ObjectReader>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = ObjectIdReader.construct(javaType0, propertyName0, (ObjectIdGenerator<?>) objectIdGenerator0, (JsonDeserializer<?>) jsonDeserializer0, (SettableBeanProperty) null, (ObjectIdResolver) simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      AnnotatedMember annotatedMember0 = objectIdValueProperty0.getMember();
      assertNull(annotatedMember0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Class<Integer> class0 = Integer.TYPE;
      SimpleType simpleType0 = SimpleType.construct(class0);
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      ObjectIdGenerator<POJONode> objectIdGenerator0 = (ObjectIdGenerator<POJONode>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      JsonDeserializer<String> jsonDeserializer0 = (JsonDeserializer<String>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = new ObjectIdReader(simpleType0, propertyName0, objectIdGenerator0, jsonDeserializer0, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      Class<Annotation> class1 = Annotation.class;
      Annotation annotation0 = objectIdValueProperty0.getAnnotation(class1);
      assertNull(annotation0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      PropertyName propertyName0 = PropertyName.construct("g5vl3DI7R", (String) null);
      ObjectIdGenerator<Object> objectIdGenerator0 = (ObjectIdGenerator<Object>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      doReturn((ObjectIdGenerator.IdKey) null).when(objectIdGenerator0).key(any());
      JsonDeserializer<ObjectReader> jsonDeserializer0 = (JsonDeserializer<ObjectReader>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      doReturn((Object) null).when(jsonDeserializer0).deserialize(any(com.fasterxml.jackson.core.JsonParser.class) , any(com.fasterxml.jackson.databind.DeserializationContext.class));
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = ObjectIdReader.construct(javaType0, propertyName0, (ObjectIdGenerator<?>) objectIdGenerator0, (JsonDeserializer<?>) jsonDeserializer0, (SettableBeanProperty) null, (ObjectIdResolver) simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser("");
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        objectIdValueProperty0.deserializeAndSet(jsonParser0, defaultDeserializationContext_Impl0, propertyName0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.ReadableObjectId", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      ObjectIdGenerator<POJONode> objectIdGenerator0 = (ObjectIdGenerator<POJONode>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      JsonDeserializer<String> jsonDeserializer0 = (JsonDeserializer<String>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = new ObjectIdReader(simpleType0, propertyName0, objectIdGenerator0, jsonDeserializer0, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      SettableBeanProperty settableBeanProperty0 = objectIdValueProperty0.withSimpleName("9#w<pZXO7~\"4K >U] c");
      assertTrue(settableBeanProperty0.hasValueDeserializer());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Class<Integer> class0 = Integer.TYPE;
      SimpleType simpleType0 = SimpleType.construct(class0);
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      ObjectIdGenerator<POJONode> objectIdGenerator0 = (ObjectIdGenerator<POJONode>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      JsonDeserializer<String> jsonDeserializer0 = (JsonDeserializer<String>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = new ObjectIdReader(simpleType0, propertyName0, objectIdGenerator0, jsonDeserializer0, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      ObjectIdValueProperty objectIdValueProperty1 = new ObjectIdValueProperty(objectIdValueProperty0, "4V,~sSj");
      assertEquals((-1), objectIdValueProperty1.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      PropertyName propertyName0 = PropertyName.construct("g5vl3DI7R", (String) null);
      ObjectIdGenerator<Object> objectIdGenerator0 = (ObjectIdGenerator<Object>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      JsonDeserializer<ObjectReader> jsonDeserializer0 = (JsonDeserializer<ObjectReader>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = ObjectIdReader.construct(javaType0, propertyName0, (ObjectIdGenerator<?>) objectIdGenerator0, (JsonDeserializer<?>) jsonDeserializer0, (SettableBeanProperty) null, (ObjectIdResolver) simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      JsonDeserializer<String> jsonDeserializer1 = (JsonDeserializer<String>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      doReturn((Object) null).when(jsonDeserializer1).getNullValue();
      ObjectIdValueProperty objectIdValueProperty1 = objectIdValueProperty0.withValueDeserializer(jsonDeserializer1);
      assertTrue(objectIdValueProperty1.isRequired());
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      Class<Annotation> class0 = Annotation.class;
      Class<ObjectReader> class1 = ObjectReader.class;
      ObjectIdGenerator.IdKey objectIdGenerator_IdKey0 = new ObjectIdGenerator.IdKey(class0, class1, propertyName0);
      ObjectIdGenerator<Object> objectIdGenerator0 = (ObjectIdGenerator<Object>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      doReturn(objectIdGenerator_IdKey0).when(objectIdGenerator0).key(any());
      JsonDeserializer<ObjectReader> jsonDeserializer0 = (JsonDeserializer<ObjectReader>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      doReturn((Object) null).when(jsonDeserializer0).deserialize(any(com.fasterxml.jackson.core.JsonParser.class) , any(com.fasterxml.jackson.databind.DeserializationContext.class));
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = ObjectIdReader.construct(javaType0, propertyName0, (ObjectIdGenerator<?>) objectIdGenerator0, (JsonDeserializer<?>) jsonDeserializer0, (SettableBeanProperty) null, (ObjectIdResolver) simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED;
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, objectIdGenerator_IdKey0, true);
      PipedWriter pipedWriter0 = new PipedWriter();
      PipedReader pipedReader0 = new PipedReader(pipedWriter0);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 0, pipedReader0, (ObjectCodec) null, charsToNameCanonicalizer0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      ReaderBasedJsonParser readerBasedJsonParser1 = (ReaderBasedJsonParser)objectIdValueProperty0.deserializeSetAndReturn(readerBasedJsonParser0, defaultDeserializationContext_Impl0, readerBasedJsonParser0);
      assertEquals(0L, readerBasedJsonParser1.getTokenCharacterOffset());
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      ObjectIdGenerator<POJONode> objectIdGenerator0 = (ObjectIdGenerator<POJONode>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      JsonDeserializer<String> jsonDeserializer0 = (JsonDeserializer<String>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = new ObjectIdReader(simpleType0, propertyName0, objectIdGenerator0, jsonDeserializer0, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      JsonDeserializer<Object> jsonDeserializer1 = objectIdReader0.getDeserializer();
      ObjectIdReader objectIdReader1 = ObjectIdReader.construct((JavaType) simpleType0, objectIdReader0.propertyName, objectIdReader0.generator, (JsonDeserializer<?>) jsonDeserializer1, (SettableBeanProperty) objectIdValueProperty0, (ObjectIdResolver) simpleObjectIdResolver0);
      ObjectIdValueProperty objectIdValueProperty1 = new ObjectIdValueProperty(objectIdReader1, propertyMetadata0);
      // Undeclared exception!
      try { 
        objectIdValueProperty1.set(objectIdValueProperty0, class0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // Should not call set() on ObjectIdProperty that has no SettableBeanProperty
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.ObjectIdValueProperty", e);
      }
  }
}
