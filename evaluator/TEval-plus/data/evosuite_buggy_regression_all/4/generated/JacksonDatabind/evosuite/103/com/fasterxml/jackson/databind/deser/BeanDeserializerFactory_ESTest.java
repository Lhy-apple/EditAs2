/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:48:49 GMT 2023
 */

package com.fasterxml.jackson.databind.deser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.cfg.MapperConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.DeserializerFactory;
import com.fasterxml.jackson.databind.deser.SettableAnyProperty;
import com.fasterxml.jackson.databind.deser.std.StdKeyDeserializers;
import com.fasterxml.jackson.databind.introspect.AnnotatedClass;
import com.fasterxml.jackson.databind.introspect.BasicBeanDescription;
import com.fasterxml.jackson.databind.introspect.ObjectIdInfo;
import com.fasterxml.jackson.databind.introspect.POJOPropertiesCollector;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.type.ResolvedRecursiveType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BeanDeserializerFactory_ESTest extends BeanDeserializerFactory_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      POJOPropertiesCollector pOJOPropertiesCollector0 = mock(POJOPropertiesCollector.class, new ViolatedAssumptionAnswer());
      doReturn((AnnotatedClass) null).when(pOJOPropertiesCollector0).getClassDef();
      doReturn((MapperConfig) null).when(pOJOPropertiesCollector0).getConfig();
      doReturn((ObjectIdInfo) null).when(pOJOPropertiesCollector0).getObjectIdInfo();
      doReturn((JavaType) null).when(pOJOPropertiesCollector0).getType();
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forSerialization(pOJOPropertiesCollector0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<SettableAnyProperty> class0 = SettableAnyProperty.class;
      // Undeclared exception!
      try { 
        beanDeserializerFactory0.createBuilderBasedDeserializer(defaultDeserializationContext_Impl0, (JavaType) null, basicBeanDescription0, class0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      JsonFactory jsonFactory0 = new JsonFactory();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0, defaultSerializerProvider_Impl0, defaultDeserializationContext_Impl0);
      Class<SettableAnyProperty> class0 = SettableAnyProperty.class;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      ObjectReader objectReader0 = objectMapper0.readerFor((JavaType) resolvedRecursiveType0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      StdKeyDeserializers stdKeyDeserializers0 = new StdKeyDeserializers();
      DeserializerFactory deserializerFactory0 = beanDeserializerFactory0.withAdditionalKeyDeserializers(stdKeyDeserializers0);
      assertNotSame(deserializerFactory0, beanDeserializerFactory0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory((DeserializerFactoryConfig) null);
      DeserializerFactory deserializerFactory0 = beanDeserializerFactory0.withConfig((DeserializerFactoryConfig) null);
      assertSame(deserializerFactory0, beanDeserializerFactory0);
  }
}
