/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:24:32 GMT 2023
 */

package com.fasterxml.jackson.databind.deser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import com.fasterxml.jackson.databind.DeserializationConfig;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.cfg.BaseSettings;
import com.fasterxml.jackson.databind.cfg.ConfigOverrides;
import com.fasterxml.jackson.databind.cfg.MapperConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerBuilder;
import com.fasterxml.jackson.databind.introspect.AnnotatedClass;
import com.fasterxml.jackson.databind.introspect.BasicBeanDescription;
import com.fasterxml.jackson.databind.introspect.ClassIntrospector;
import com.fasterxml.jackson.databind.introspect.ObjectIdInfo;
import com.fasterxml.jackson.databind.introspect.POJOPropertiesCollector;
import com.fasterxml.jackson.databind.introspect.SimpleMixInResolver;
import com.fasterxml.jackson.databind.jsontype.impl.StdSubtypeResolver;
import com.fasterxml.jackson.databind.util.RootNameLookup;
import java.lang.reflect.InvocationTargetException;
import java.sql.SQLNonTransientException;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BeanDeserializerBase_ESTest extends BeanDeserializerBase_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      SQLNonTransientException sQLNonTransientException0 = new SQLNonTransientException();
      InvocationTargetException invocationTargetException0 = new InvocationTargetException(sQLNonTransientException0);
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(invocationTargetException0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      POJOPropertiesCollector pOJOPropertiesCollector0 = mock(POJOPropertiesCollector.class, new ViolatedAssumptionAnswer());
      doReturn((AnnotatedClass) null).when(pOJOPropertiesCollector0).getClassDef();
      doReturn((MapperConfig) null).when(pOJOPropertiesCollector0).getConfig();
      doReturn((ObjectIdInfo) null).when(pOJOPropertiesCollector0).getObjectIdInfo();
      doReturn((JavaType) null).when(pOJOPropertiesCollector0).getType();
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forSerialization(pOJOPropertiesCollector0);
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      SimpleMixInResolver simpleMixInResolver0 = new SimpleMixInResolver((ClassIntrospector.MixInResolver) null);
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      ConfigOverrides configOverrides0 = new ConfigOverrides();
      DeserializationConfig deserializationConfig0 = new DeserializationConfig((BaseSettings) null, stdSubtypeResolver0, simpleMixInResolver0, rootNameLookup0, configOverrides0);
      BeanDeserializerBuilder beanDeserializerBuilder0 = new BeanDeserializerBuilder(basicBeanDescription0, deserializationConfig0);
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(beanDeserializerBuilder0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      SimpleMixInResolver simpleMixInResolver0 = new SimpleMixInResolver((ClassIntrospector.MixInResolver) null);
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectMapper objectMapper1 = objectMapper0.enableDefaultTyping();
      ObjectReader objectReader0 = objectMapper1.readerForUpdating(simpleMixInResolver0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      ConfigOverrides configOverrides0 = new ConfigOverrides();
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(configOverrides0);
      assertNotNull(objectReader0);
  }
}
