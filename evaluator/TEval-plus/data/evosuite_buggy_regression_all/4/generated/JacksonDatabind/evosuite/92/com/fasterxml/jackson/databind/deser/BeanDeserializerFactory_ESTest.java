/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:47:37 GMT 2023
 */

package com.fasterxml.jackson.databind.deser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import com.fasterxml.jackson.databind.BeanDescription;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.cfg.MapperConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerBuilder;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.BeanDeserializerModifier;
import com.fasterxml.jackson.databind.deser.BuilderBasedDeserializer;
import com.fasterxml.jackson.databind.deser.CreatorProperty;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.DeserializerFactory;
import com.fasterxml.jackson.databind.ext.CoreXMLDeserializers;
import com.fasterxml.jackson.databind.introspect.AnnotatedClass;
import com.fasterxml.jackson.databind.introspect.BasicBeanDescription;
import com.fasterxml.jackson.databind.introspect.ObjectIdInfo;
import com.fasterxml.jackson.databind.introspect.POJOPropertiesCollector;
import com.fasterxml.jackson.databind.type.SimpleType;
import java.time.chrono.MinguoEra;
import java.util.Map;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BeanDeserializerFactory_ESTest extends BeanDeserializerFactory_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      Class<MinguoEra> class0 = MinguoEra.class;
      // Undeclared exception!
      try { 
        beanDeserializerFactory0.isPotentialBeanType(class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not deserialize Class java.time.chrono.MinguoEra (of type enum) as a Bean
         //
         verifyException("com.fasterxml.jackson.databind.deser.BeanDeserializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      Class<CoreXMLDeserializers.Std> class0 = CoreXMLDeserializers.Std.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      POJOPropertiesCollector pOJOPropertiesCollector0 = mock(POJOPropertiesCollector.class, new ViolatedAssumptionAnswer());
      doReturn((AnnotatedClass) null).when(pOJOPropertiesCollector0).getClassDef();
      doReturn((MapperConfig) null).when(pOJOPropertiesCollector0).getConfig();
      doReturn((ObjectIdInfo) null).when(pOJOPropertiesCollector0).getObjectIdInfo();
      doReturn((JavaType) null).when(pOJOPropertiesCollector0).getType();
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forSerialization(pOJOPropertiesCollector0);
      // Undeclared exception!
      try { 
        beanDeserializerFactory0.instance.createBuilderBasedDeserializer(defaultDeserializationContext_Impl0, simpleType0, basicBeanDescription0, class0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      DeserializerFactory deserializerFactory0 = beanDeserializerFactory0.withConfig(deserializerFactoryConfig0);
      assertNotSame(deserializerFactory0, beanDeserializerFactory0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory((DeserializerFactoryConfig) null);
      DeserializerFactory deserializerFactory0 = beanDeserializerFactory0.withConfig((DeserializerFactoryConfig) null);
      assertSame(deserializerFactory0, beanDeserializerFactory0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerModifier beanDeserializerModifier0 = mock(BeanDeserializerModifier.class, new ViolatedAssumptionAnswer());
      doReturn((JsonDeserializer) null).when(beanDeserializerModifier0).modifyDeserializer(any(com.fasterxml.jackson.databind.DeserializationConfig.class) , any(com.fasterxml.jackson.databind.BeanDescription.class) , any(com.fasterxml.jackson.databind.JsonDeserializer.class));
      DeserializerFactoryConfig deserializerFactoryConfig1 = deserializerFactoryConfig0.withDeserializerModifier(beanDeserializerModifier0);
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig1);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        beanDeserializerFactory0.createBeanDeserializer(defaultDeserializationContext_Impl0, simpleType0, (BeanDescription) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.BasicDeserializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      MapperFeature mapperFeature0 = MapperFeature.USE_GETTERS_AS_SETTERS;
      objectMapper0.configure(mapperFeature0, false);
      Class<BuilderBasedDeserializer> class0 = BuilderBasedDeserializer.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      MapperFeature[] mapperFeatureArray0 = new MapperFeature[2];
      MapperFeature mapperFeature0 = MapperFeature.AUTO_DETECT_GETTERS;
      mapperFeatureArray0[0] = mapperFeature0;
      mapperFeatureArray0[1] = mapperFeatureArray0[0];
      ObjectMapper objectMapper1 = objectMapper0.disable(mapperFeatureArray0);
      Class<ObjectIdGenerators.StringIdGenerator> class0 = ObjectIdGenerators.StringIdGenerator.class;
      ObjectReader objectReader0 = objectMapper1.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<BuilderBasedDeserializer> class0 = BuilderBasedDeserializer.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
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
      assertEquals(0, defaultDeserializationContext_Impl0.getDeserializationFeatures());
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      objectMapper0.enableDefaultTyping();
      Class<CreatorProperty> class0 = CreatorProperty.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }
}