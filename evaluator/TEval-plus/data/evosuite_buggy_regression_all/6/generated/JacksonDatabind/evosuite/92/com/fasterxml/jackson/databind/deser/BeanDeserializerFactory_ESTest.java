/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:29:25 GMT 2023
 */

package com.fasterxml.jackson.databind.deser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.fasterxml.jackson.databind.BeanDescription;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.Module;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.cfg.MapperConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerBuilder;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.DeserializerFactory;
import com.fasterxml.jackson.databind.ext.CoreXMLDeserializers;
import com.fasterxml.jackson.databind.introspect.AnnotatedClass;
import com.fasterxml.jackson.databind.introspect.BasicBeanDescription;
import com.fasterxml.jackson.databind.introspect.ObjectIdInfo;
import com.fasterxml.jackson.databind.introspect.POJOPropertiesCollector;
import java.util.Map;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BeanDeserializerFactory_ESTest extends BeanDeserializerFactory_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DeserializerFactory deserializerFactory0 = beanDeserializerFactory0.withConfig(deserializerFactoryConfig0);
      assertSame(deserializerFactory0, beanDeserializerFactory0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<Throwable> class0 = Throwable.class;
      // Undeclared exception!
      try { 
        beanDeserializerFactory0.createBuilderBasedDeserializer(defaultDeserializationContext_Impl0, (JavaType) null, (BeanDescription) null, class0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      DeserializerFactory deserializerFactory0 = beanDeserializerFactory0.withConfig(deserializerFactoryConfig0);
      assertNotSame(deserializerFactory0, beanDeserializerFactory0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<Throwable> class0 = Throwable.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<Integer> class0 = Integer.TYPE;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<CoreXMLDeserializers.Std> class0 = CoreXMLDeserializers.Std.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      MapperFeature[] mapperFeatureArray0 = new MapperFeature[9];
      MapperFeature mapperFeature0 = MapperFeature.USE_GETTERS_AS_SETTERS;
      mapperFeatureArray0[0] = mapperFeature0;
      mapperFeatureArray0[1] = mapperFeature0;
      mapperFeatureArray0[2] = mapperFeatureArray0[0];
      mapperFeatureArray0[3] = mapperFeatureArray0[0];
      mapperFeatureArray0[4] = mapperFeatureArray0[3];
      mapperFeatureArray0[5] = mapperFeatureArray0[1];
      mapperFeatureArray0[6] = mapperFeatureArray0[0];
      mapperFeatureArray0[7] = mapperFeatureArray0[1];
      MapperFeature mapperFeature1 = MapperFeature.DEFAULT_VIEW_INCLUSION;
      mapperFeatureArray0[8] = mapperFeature1;
      ObjectMapper objectMapper1 = objectMapper0.disable(mapperFeatureArray0);
      Class<Module> class0 = Module.class;
      ObjectReader objectReader0 = objectMapper1.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      POJOPropertiesCollector pOJOPropertiesCollector0 = mock(POJOPropertiesCollector.class, new ViolatedAssumptionAnswer());
      doReturn((AnnotatedClass) null).when(pOJOPropertiesCollector0).getClassDef();
      doReturn((MapperConfig) null).when(pOJOPropertiesCollector0).getConfig();
      doReturn((Map) null).when(pOJOPropertiesCollector0).getInjectables();
      doReturn((ObjectIdInfo) null).when(pOJOPropertiesCollector0).getObjectIdInfo();
      doReturn((JavaType) null).when(pOJOPropertiesCollector0).getType();
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forDeserialization(pOJOPropertiesCollector0);
      beanDeserializerFactory0.addInjectables(defaultDeserializationContext_Impl0, basicBeanDescription0, (BeanDeserializerBuilder) null);
      assertNull(basicBeanDescription0.findClassDescription());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      MapperFeature[] mapperFeatureArray0 = new MapperFeature[9];
      MapperFeature mapperFeature0 = MapperFeature.USE_GETTERS_AS_SETTERS;
      mapperFeatureArray0[0] = mapperFeature0;
      MapperFeature mapperFeature1 = MapperFeature.CAN_OVERRIDE_ACCESS_MODIFIERS;
      mapperFeatureArray0[1] = mapperFeature1;
      mapperFeatureArray0[2] = mapperFeatureArray0[0];
      mapperFeatureArray0[3] = mapperFeatureArray0[0];
      mapperFeatureArray0[4] = mapperFeature1;
      mapperFeatureArray0[5] = mapperFeatureArray0[2];
      mapperFeatureArray0[6] = mapperFeature0;
      mapperFeatureArray0[7] = mapperFeatureArray0[0];
      mapperFeatureArray0[8] = mapperFeatureArray0[1];
      ObjectMapper objectMapper1 = objectMapper0.disable(mapperFeatureArray0);
      Class<Module> class0 = Module.class;
      ObjectReader objectReader0 = objectMapper1.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      objectMapper0.enableDefaultTyping();
      Class<Module> class0 = Module.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      Class<JsonTypeInfo.As> class0 = JsonTypeInfo.As.class;
      // Undeclared exception!
      try { 
        beanDeserializerFactory0.isPotentialBeanType(class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not deserialize Class com.fasterxml.jackson.annotation.JsonTypeInfo$As (of type enum) as a Bean
         //
         verifyException("com.fasterxml.jackson.databind.deser.BeanDeserializerFactory", e);
      }
  }
}