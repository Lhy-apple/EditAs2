/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:04:26 GMT 2023
 */

package com.fasterxml.jackson.databind.deser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.databind.BeanDescription;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.cfg.MapperConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerBuilder;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.BeanDeserializerModifier;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.DeserializerFactory;
import com.fasterxml.jackson.databind.ext.CoreXMLDeserializers;
import com.fasterxml.jackson.databind.introspect.AnnotatedClass;
import com.fasterxml.jackson.databind.introspect.AnnotatedMember;
import com.fasterxml.jackson.databind.introspect.BasicBeanDescription;
import com.fasterxml.jackson.databind.introspect.ObjectIdInfo;
import com.fasterxml.jackson.databind.introspect.POJOPropertiesCollector;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.util.HashMap;
import java.util.Map;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.testdata.EvoSuiteFile;
import org.evosuite.runtime.testdata.FileSystemHandling;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BeanDeserializerFactory_ESTest extends BeanDeserializerFactory_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      HashMap<Object, AnnotatedMember> hashMap0 = new HashMap<Object, AnnotatedMember>();
      POJOPropertiesCollector pOJOPropertiesCollector0 = mock(POJOPropertiesCollector.class, new ViolatedAssumptionAnswer());
      doReturn((AnnotatedClass) null).when(pOJOPropertiesCollector0).getClassDef();
      doReturn((MapperConfig) null).when(pOJOPropertiesCollector0).getConfig();
      doReturn(hashMap0).when(pOJOPropertiesCollector0).getInjectables();
      doReturn((ObjectIdInfo) null).when(pOJOPropertiesCollector0).getObjectIdInfo();
      doReturn((JavaType) null).when(pOJOPropertiesCollector0).getType();
      Object object0 = new Object();
      hashMap0.put(object0, (AnnotatedMember) null);
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forDeserialization(pOJOPropertiesCollector0);
      BeanDeserializerBuilder beanDeserializerBuilder0 = beanDeserializerFactory0.constructBeanDeserializerBuilder(defaultDeserializationContext_Impl0, basicBeanDescription0);
      // Undeclared exception!
      try { 
        beanDeserializerFactory0.addInjectables(defaultDeserializationContext_Impl0, basicBeanDescription0, beanDeserializerBuilder0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.BeanDeserializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Object[] objectArray0 = new Object[5];
      JsonMappingException jsonMappingException0 = defaultSerializerProvider_Impl0.mappingException("com.fasterxml.jacson.databind.ObjectMapper", objectArray0);
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(jsonMappingException0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      Class<CoreXMLDeserializers.Std> class0 = CoreXMLDeserializers.Std.class;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JavaType javaType0 = TypeFactory.unknownType();
      // Undeclared exception!
      try { 
        beanDeserializerFactory0.createBuilderBasedDeserializer(defaultDeserializationContext_Impl0, javaType0, (BeanDescription) null, class0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      DeserializerFactory deserializerFactory0 = beanDeserializerFactory0.withConfig(deserializerFactoryConfig0);
      assertNotSame(deserializerFactory0, beanDeserializerFactory0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(jsonFactory0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      deserializerFactoryConfig0.abstractTypeResolvers();
      BeanDeserializerModifier beanDeserializerModifier0 = mock(BeanDeserializerModifier.class, new ViolatedAssumptionAnswer());
      FileSystemHandling.appendLineToFile((EvoSuiteFile) null, "~");
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DeserializerFactory deserializerFactory0 = beanDeserializerFactory0.withConfig(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(deserializerFactory0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl1 = new DefaultDeserializationContext.Impl(defaultDeserializationContext_Impl0, beanDeserializerFactory0);
      POJOPropertiesCollector pOJOPropertiesCollector0 = mock(POJOPropertiesCollector.class, new ViolatedAssumptionAnswer());
      doReturn((AnnotatedClass) null).when(pOJOPropertiesCollector0).getClassDef();
      doReturn((MapperConfig) null).when(pOJOPropertiesCollector0).getConfig();
      doReturn((Map) null).when(pOJOPropertiesCollector0).getInjectables();
      doReturn((ObjectIdInfo) null).when(pOJOPropertiesCollector0).getObjectIdInfo();
      doReturn((JavaType) null).when(pOJOPropertiesCollector0).getType();
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forDeserialization(pOJOPropertiesCollector0);
      BeanDeserializerBuilder beanDeserializerBuilder0 = beanDeserializerFactory0.constructBeanDeserializerBuilder(defaultDeserializationContext_Impl1, basicBeanDescription0);
      beanDeserializerFactory0.addInjectables(defaultDeserializationContext_Impl1, basicBeanDescription0, beanDeserializerBuilder0);
      assertSame(deserializerFactory0, beanDeserializerFactory0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      // Undeclared exception!
      try { 
        beanDeserializerFactory0.constructAnySetter((DeserializationContext) null, (BeanDescription) null, (AnnotatedMember) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.BasicDeserializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
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