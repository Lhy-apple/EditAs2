/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 19:14:51 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.fasterxml.jackson.annotation.PropertyAccessor;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.DeserializationConfig;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.cfg.BaseSettings;
import com.fasterxml.jackson.databind.cfg.ConfigOverrides;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.ValueInstantiator;
import com.fasterxml.jackson.databind.deser.std.AtomicReferenceDeserializer;
import com.fasterxml.jackson.databind.introspect.ClassIntrospector;
import com.fasterxml.jackson.databind.introspect.SimpleMixInResolver;
import com.fasterxml.jackson.databind.jsontype.TypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.TypeIdResolver;
import com.fasterxml.jackson.databind.jsontype.impl.AsPropertyTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.StdSubtypeResolver;
import com.fasterxml.jackson.databind.node.DoubleNode;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.ReferenceType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import com.fasterxml.jackson.databind.util.RootNameLookup;
import java.util.concurrent.atomic.AtomicReference;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class AtomicReferenceDeserializer_ESTest extends AtomicReferenceDeserializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Class<PropertyAccessor> class0 = PropertyAccessor.class;
      JavaType javaType0 = TypeFactory.unknownType();
      ReferenceType referenceType0 = ReferenceType.upgradeFrom(javaType0, javaType0);
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      AtomicReferenceDeserializer atomicReferenceDeserializer0 = new AtomicReferenceDeserializer(referenceType0, valueInstantiator_Base0, (TypeDeserializer) null, (JsonDeserializer<?>) null);
      AtomicReferenceDeserializer atomicReferenceDeserializer1 = atomicReferenceDeserializer0.withResolved((TypeDeserializer) null, (JsonDeserializer<?>) null);
      assertNotSame(atomicReferenceDeserializer0, atomicReferenceDeserializer1);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Class<PropertyAccessor> class0 = PropertyAccessor.class;
      JavaType javaType0 = TypeFactory.unknownType();
      MapType mapType0 = MapType.construct(class0, (TypeBindings) null, javaType0, (JavaType[]) null, javaType0, javaType0);
      Class<Object> class1 = Object.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class1);
      JsonTypeInfo.As jsonTypeInfo_As0 = JsonTypeInfo.As.WRAPPER_OBJECT;
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(mapType0, (TypeIdResolver) null, "com.fasterxml.jackson.databind.ser.std.NumberSerializer", false, mapType0, jsonTypeInfo_As0);
      JsonDeserializer<DoubleNode> jsonDeserializer0 = (JsonDeserializer<DoubleNode>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      AtomicReferenceDeserializer atomicReferenceDeserializer0 = new AtomicReferenceDeserializer(mapType0, valueInstantiator_Base0, asPropertyTypeDeserializer0, jsonDeserializer0);
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      SimpleMixInResolver simpleMixInResolver0 = new SimpleMixInResolver((ClassIntrospector.MixInResolver) null);
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      ConfigOverrides configOverrides0 = new ConfigOverrides();
      DeserializationConfig deserializationConfig0 = new DeserializationConfig((BaseSettings) null, stdSubtypeResolver0, simpleMixInResolver0, rootNameLookup0, configOverrides0);
      Boolean boolean0 = atomicReferenceDeserializer0.supportsUpdate(deserializationConfig0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      ReferenceType referenceType0 = ReferenceType.upgradeFrom(javaType0, javaType0);
      Class<Integer> class0 = Integer.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      AtomicReferenceDeserializer atomicReferenceDeserializer0 = new AtomicReferenceDeserializer(referenceType0, valueInstantiator_Base0, (TypeDeserializer) null, (JsonDeserializer<?>) null);
      AtomicReference<Object> atomicReference0 = new AtomicReference<Object>();
      Object object0 = atomicReferenceDeserializer0.getReferenced(atomicReference0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Class<PropertyAccessor> class0 = PropertyAccessor.class;
      JavaType javaType0 = TypeFactory.unknownType();
      MapType mapType0 = MapType.construct(class0, (TypeBindings) null, javaType0, (JavaType[]) null, javaType0, javaType0);
      Class<Object> class1 = Object.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class1);
      JsonTypeInfo.As jsonTypeInfo_As0 = JsonTypeInfo.As.WRAPPER_OBJECT;
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(mapType0, (TypeIdResolver) null, "com.fasterxml.jackson.databind.ser.std.NumberSerializer", false, mapType0, jsonTypeInfo_As0);
      JsonDeserializer<DoubleNode> jsonDeserializer0 = (JsonDeserializer<DoubleNode>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      AtomicReferenceDeserializer atomicReferenceDeserializer0 = new AtomicReferenceDeserializer(mapType0, valueInstantiator_Base0, asPropertyTypeDeserializer0, jsonDeserializer0);
      Object object0 = atomicReferenceDeserializer0.getEmptyValue((DeserializationContext) null);
      assertEquals("null", object0.toString());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      ReferenceType referenceType0 = ReferenceType.upgradeFrom(javaType0, javaType0);
      Class<Integer> class0 = Integer.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      AtomicReferenceDeserializer atomicReferenceDeserializer0 = new AtomicReferenceDeserializer(referenceType0, valueInstantiator_Base0, (TypeDeserializer) null, (JsonDeserializer<?>) null);
      AtomicReference<Object> atomicReference0 = new AtomicReference<Object>();
      AtomicReference<Object> atomicReference1 = atomicReferenceDeserializer0.updateReference(atomicReference0, (Object) javaType0);
      assertSame(atomicReference1, atomicReference0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      BeanProperty.Bogus beanProperty_Bogus0 = new BeanProperty.Bogus();
      JavaType javaType0 = beanProperty_Bogus0.getType();
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(javaType0);
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(javaType0, (TypeIdResolver) null, "String \"%s\"", true, javaType0);
      JsonDeserializer<Integer> jsonDeserializer0 = (JsonDeserializer<Integer>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      AtomicReferenceDeserializer atomicReferenceDeserializer0 = new AtomicReferenceDeserializer(javaType0, valueInstantiator_Base0, asPropertyTypeDeserializer0, jsonDeserializer0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      AtomicReference<Object> atomicReference0 = atomicReferenceDeserializer0.getNullValue((DeserializationContext) defaultDeserializationContext_Impl0);
      assertEquals("null", atomicReference0.toString());
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      BeanProperty.Bogus beanProperty_Bogus0 = new BeanProperty.Bogus();
      JavaType javaType0 = beanProperty_Bogus0.getType();
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(javaType0);
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(javaType0, (TypeIdResolver) null, "String \"%s\"", true, javaType0);
      JsonDeserializer<Integer> jsonDeserializer0 = (JsonDeserializer<Integer>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      AtomicReferenceDeserializer atomicReferenceDeserializer0 = new AtomicReferenceDeserializer(javaType0, valueInstantiator_Base0, asPropertyTypeDeserializer0, jsonDeserializer0);
      AtomicReference<Object> atomicReference0 = atomicReferenceDeserializer0.referenceValue((Object) null);
      assertEquals("null", atomicReference0.toString());
  }
}
