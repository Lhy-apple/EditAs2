/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 03:12:39 GMT 2023
 */

package com.fasterxml.jackson.databind.deser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.databind.BeanDescription;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.cfg.MapperConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerBuilder;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.DeserializerFactory;
import com.fasterxml.jackson.databind.ext.PathDeserializer;
import com.fasterxml.jackson.databind.introspect.AnnotatedClass;
import com.fasterxml.jackson.databind.introspect.BasicBeanDescription;
import com.fasterxml.jackson.databind.introspect.ObjectIdInfo;
import com.fasterxml.jackson.databind.introspect.POJOPropertiesCollector;
import com.fasterxml.jackson.databind.introspect.VirtualAnnotatedMember;
import java.sql.SQLTimeoutException;
import java.time.temporal.ChronoField;
import java.util.Map;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BeanDeserializerFactory_ESTest extends BeanDeserializerFactory_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(jsonFactory0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      Class<VirtualAnnotatedMember> class0 = VirtualAnnotatedMember.class;
      // Undeclared exception!
      try { 
        beanDeserializerFactory0.createBuilderBasedDeserializer((DeserializationContext) null, (JavaType) null, (BeanDescription) null, class0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.BeanDeserializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DeserializerFactory deserializerFactory0 = beanDeserializerFactory0.withConfig((DeserializerFactoryConfig) null);
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
      ObjectMapper objectMapper0 = new ObjectMapper();
      SQLTimeoutException sQLTimeoutException0 = new SQLTimeoutException();
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(sQLTimeoutException0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      MapperFeature[] mapperFeatureArray0 = new MapperFeature[4];
      MapperFeature mapperFeature0 = MapperFeature.DEFAULT_VIEW_INCLUSION;
      mapperFeatureArray0[0] = mapperFeature0;
      MapperFeature mapperFeature1 = MapperFeature.AUTO_DETECT_GETTERS;
      mapperFeatureArray0[1] = mapperFeature1;
      mapperFeatureArray0[2] = mapperFeature1;
      mapperFeatureArray0[3] = mapperFeatureArray0[0];
      objectMapper0.disable(mapperFeatureArray0);
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(objectMapper0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      PathDeserializer pathDeserializer0 = new PathDeserializer();
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(pathDeserializer0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
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
  public void test8()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      Class<ChronoField> class0 = ChronoField.class;
      // Undeclared exception!
      try { 
        beanDeserializerFactory0.isPotentialBeanType(class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not deserialize Class java.time.temporal.ChronoField (of type enum) as a Bean
         //
         verifyException("com.fasterxml.jackson.databind.deser.BeanDeserializerFactory", e);
      }
  }
}