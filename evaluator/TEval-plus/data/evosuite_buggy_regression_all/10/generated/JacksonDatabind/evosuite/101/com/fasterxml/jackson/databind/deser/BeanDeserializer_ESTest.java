/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:42:34 GMT 2023
 */

package com.fasterxml.jackson.databind.deser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.cfg.MapperConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializer;
import com.fasterxml.jackson.databind.deser.BeanDeserializerBase;
import com.fasterxml.jackson.databind.deser.BeanDeserializerBuilder;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.SettableBeanProperty;
import com.fasterxml.jackson.databind.deser.impl.BeanPropertyMap;
import com.fasterxml.jackson.databind.introspect.AnnotatedClass;
import com.fasterxml.jackson.databind.introspect.BasicBeanDescription;
import com.fasterxml.jackson.databind.introspect.ObjectIdInfo;
import com.fasterxml.jackson.databind.introspect.POJOPropertiesCollector;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Vector;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BeanDeserializer_ESTest extends BeanDeserializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      BeanDeserializer beanDeserializer0 = null;
      try {
        beanDeserializer0 = new BeanDeserializer((BeanDeserializerBase) null, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.BeanDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      BeanDeserializer beanDeserializer0 = null;
      try {
        beanDeserializer0 = new BeanDeserializer((BeanDeserializerBase) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.BeanDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      POJOPropertiesCollector pOJOPropertiesCollector0 = mock(POJOPropertiesCollector.class, new ViolatedAssumptionAnswer());
      doReturn((AnnotatedClass) null).when(pOJOPropertiesCollector0).getClassDef();
      doReturn((MapperConfig) null).when(pOJOPropertiesCollector0).getConfig();
      doReturn((ObjectIdInfo) null).when(pOJOPropertiesCollector0).getObjectIdInfo();
      doReturn((JavaType) null).when(pOJOPropertiesCollector0).getType();
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forDeserialization(pOJOPropertiesCollector0);
      BeanDeserializerBuilder beanDeserializerBuilder0 = beanDeserializerFactory0.constructBeanDeserializerBuilder(defaultDeserializationContext_Impl0, basicBeanDescription0);
      Vector<SettableBeanProperty> vector0 = new Vector<SettableBeanProperty>();
      LinkedHashMap<String, List<PropertyName>> linkedHashMap0 = new LinkedHashMap<String, List<PropertyName>>(0);
      BeanPropertyMap beanPropertyMap0 = BeanPropertyMap.construct((Collection<SettableBeanProperty>) vector0, false, (Map<String, List<PropertyName>>) linkedHashMap0);
      Map<String, SettableBeanProperty> map0 = beanDeserializerBuilder0._properties;
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      BeanDeserializer beanDeserializer0 = null;
      try {
        beanDeserializer0 = new BeanDeserializer(beanDeserializerBuilder0, basicBeanDescription0, beanPropertyMap0, map0, linkedHashSet0, true, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.BeanDeserializerBase", e);
      }
  }
}