/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:03:20 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.ObjectIdGenerator;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import com.fasterxml.jackson.annotation.ObjectIdResolver;
import com.fasterxml.jackson.annotation.SimpleObjectIdResolver;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.PropertyMetadata;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.SettableBeanProperty;
import com.fasterxml.jackson.databind.deser.impl.ObjectIdReader;
import com.fasterxml.jackson.databind.deser.impl.ObjectIdValueProperty;
import com.fasterxml.jackson.databind.introspect.AnnotatedMember;
import com.fasterxml.jackson.databind.node.NullNode;
import com.fasterxml.jackson.databind.type.ReferenceType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.io.SequenceInputStream;
import java.lang.annotation.Annotation;
import java.time.chrono.ThaiBuddhistEra;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ObjectIdValueProperty_ESTest extends ObjectIdValueProperty_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      PropertyName propertyName0 = PropertyName.construct("", "");
      ObjectIdGenerators.IntSequenceGenerator objectIdGenerators_IntSequenceGenerator0 = new ObjectIdGenerators.IntSequenceGenerator();
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = new ObjectIdReader(javaType0, propertyName0, objectIdGenerators_IntSequenceGenerator0, (JsonDeserializer<?>) null, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      AnnotatedMember annotatedMember0 = objectIdValueProperty0.getMember();
      assertNull(annotatedMember0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      PropertyName propertyName0 = PropertyName.construct("", "");
      ObjectIdGenerators.IntSequenceGenerator objectIdGenerators_IntSequenceGenerator0 = new ObjectIdGenerators.IntSequenceGenerator();
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = new ObjectIdReader(javaType0, propertyName0, objectIdGenerators_IntSequenceGenerator0, (JsonDeserializer<?>) null, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      ObjectIdValueProperty objectIdValueProperty1 = objectIdValueProperty0.withName(objectIdReader0.propertyName);
      assertFalse(objectIdValueProperty1.hasValueDeserializer());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      PropertyName propertyName0 = PropertyName.construct("", "");
      ObjectIdGenerators.IntSequenceGenerator objectIdGenerators_IntSequenceGenerator0 = new ObjectIdGenerators.IntSequenceGenerator();
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = new ObjectIdReader(javaType0, propertyName0, objectIdGenerators_IntSequenceGenerator0, (JsonDeserializer<?>) null, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      Integer integer0 = new Integer(133);
      // Undeclared exception!
      try { 
        objectIdValueProperty0.set("", integer0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // Should not call set() on ObjectIdProperty that has no SettableBeanProperty
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.ObjectIdValueProperty", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      PropertyName propertyName0 = PropertyName.construct("", "");
      ObjectIdGenerators.IntSequenceGenerator objectIdGenerators_IntSequenceGenerator0 = new ObjectIdGenerators.IntSequenceGenerator();
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = new ObjectIdReader(javaType0, propertyName0, objectIdGenerators_IntSequenceGenerator0, (JsonDeserializer<?>) null, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(false, "", (Integer) null, "");
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      Class<Annotation> class0 = Annotation.class;
      Annotation annotation0 = objectIdValueProperty0.getAnnotation(class0);
      assertNull(annotation0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      PropertyName propertyName0 = PropertyName.construct("", "");
      ObjectIdGenerators.IntSequenceGenerator objectIdGenerators_IntSequenceGenerator0 = new ObjectIdGenerators.IntSequenceGenerator();
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = new ObjectIdReader(javaType0, propertyName0, objectIdGenerators_IntSequenceGenerator0, (JsonDeserializer<?>) null, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(false, "", (Integer) null, "");
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        objectIdValueProperty0.deserializeAndSet((JsonParser) null, defaultDeserializationContext_Impl0, propertyMetadata0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.ObjectIdValueProperty", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      PropertyName propertyName0 = PropertyName.construct("", "");
      ObjectIdGenerators.IntSequenceGenerator objectIdGenerators_IntSequenceGenerator0 = new ObjectIdGenerators.IntSequenceGenerator();
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      ObjectIdReader objectIdReader0 = new ObjectIdReader(javaType0, propertyName0, objectIdGenerators_IntSequenceGenerator0, (JsonDeserializer<?>) null, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      ObjectIdValueProperty objectIdValueProperty1 = objectIdValueProperty0.withValueDeserializer((JsonDeserializer<?>) null);
      assertNotSame(objectIdValueProperty1, objectIdValueProperty0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      PropertyName propertyName0 = PropertyName.construct("I<U", "I<U");
      ObjectIdGenerators.IntSequenceGenerator objectIdGenerators_IntSequenceGenerator0 = new ObjectIdGenerators.IntSequenceGenerator();
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      JsonDeserializer<String> jsonDeserializer0 = (JsonDeserializer<String>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      doReturn("I<U").when(jsonDeserializer0).deserialize(any(com.fasterxml.jackson.core.JsonParser.class) , any(com.fasterxml.jackson.databind.DeserializationContext.class));
      ObjectIdReader objectIdReader0 = new ObjectIdReader(javaType0, propertyName0, objectIdGenerators_IntSequenceGenerator0, jsonDeserializer0, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      SimpleType simpleType0 = (SimpleType)objectIdValueProperty0.deserializeSetAndReturn((JsonParser) null, defaultDeserializationContext_Impl0, javaType0);
      assertFalse(simpleType0.isFinal());
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      ObjectIdGenerators.IntSequenceGenerator objectIdGenerators_IntSequenceGenerator0 = new ObjectIdGenerators.IntSequenceGenerator();
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      JsonDeserializer<SequenceInputStream> jsonDeserializer0 = (JsonDeserializer<SequenceInputStream>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      doReturn((Object) null).when(jsonDeserializer0).deserialize(any(com.fasterxml.jackson.core.JsonParser.class) , any(com.fasterxml.jackson.databind.DeserializationContext.class));
      ObjectIdReader objectIdReader0 = new ObjectIdReader(javaType0, propertyName0, objectIdGenerators_IntSequenceGenerator0, jsonDeserializer0, (SettableBeanProperty) null, simpleObjectIdResolver0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED;
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, propertyMetadata0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Object object0 = objectIdValueProperty0.deserializeSetAndReturn((JsonParser) null, defaultDeserializationContext_Impl0, propertyName0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      PropertyName propertyName0 = PropertyName.construct("", "");
      ObjectIdGenerators.IntSequenceGenerator objectIdGenerators_IntSequenceGenerator0 = new ObjectIdGenerators.IntSequenceGenerator();
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      Class<Object> class0 = Object.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      JavaType[] javaTypeArray0 = new JavaType[7];
      javaTypeArray0[6] = javaType0;
      ReferenceType referenceType0 = ReferenceType.construct(class0, typeBindings0, javaType0, javaTypeArray0, javaTypeArray0[6]);
      ObjectIdReader objectIdReader0 = ObjectIdReader.construct((JavaType) referenceType0, propertyName0, (ObjectIdGenerator<?>) objectIdGenerators_IntSequenceGenerator0, (JsonDeserializer<?>) null, (SettableBeanProperty) null, (ObjectIdResolver) simpleObjectIdResolver0);
      ObjectIdValueProperty objectIdValueProperty0 = new ObjectIdValueProperty(objectIdReader0, (PropertyMetadata) null);
      ObjectIdReader objectIdReader1 = new ObjectIdReader(javaType0, propertyName0, objectIdGenerators_IntSequenceGenerator0, (JsonDeserializer<?>) null, objectIdValueProperty0, objectIdReader0.resolver);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED;
      ObjectIdValueProperty objectIdValueProperty1 = new ObjectIdValueProperty(objectIdReader1, propertyMetadata0);
      NullNode nullNode0 = NullNode.getInstance();
      ThaiBuddhistEra thaiBuddhistEra0 = ThaiBuddhistEra.BE;
      // Undeclared exception!
      try { 
        objectIdValueProperty1.setAndReturn(nullNode0, thaiBuddhistEra0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // Should not call set() on ObjectIdProperty that has no SettableBeanProperty
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.ObjectIdValueProperty", e);
      }
  }
}