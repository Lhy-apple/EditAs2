/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:40:46 GMT 2023
 */

package com.fasterxml.jackson.databind.jsontype.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.PropertyMetadata;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.introspect.TypeResolutionContext;
import com.fasterxml.jackson.databind.introspect.VirtualAnnotatedMember;
import com.fasterxml.jackson.databind.jsontype.TypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.TypeIdResolver;
import com.fasterxml.jackson.databind.jsontype.impl.AsPropertyTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver;
import com.fasterxml.jackson.databind.jsontype.impl.MinimalClassNameIdResolver;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import com.fasterxml.jackson.databind.util.Annotations;
import com.fasterxml.jackson.databind.util.TokenBuffer;
import java.io.InputStream;
import java.util.Map;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class AsPropertyTypeDeserializer_ESTest extends AsPropertyTypeDeserializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      MinimalClassNameIdResolver minimalClassNameIdResolver0 = new MinimalClassNameIdResolver(javaType0, typeFactory0);
      Class<Map> class0 = Map.class;
      Class<String> class1 = String.class;
      MapType mapType0 = typeFactory0.constructMapType(class0, class1, class0);
      JsonTypeInfo.As jsonTypeInfo_As0 = JsonTypeInfo.As.EXISTING_PROPERTY;
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(javaType0, minimalClassNameIdResolver0, "TDv3\"RMFWmK*l-k5Q", false, mapType0, jsonTypeInfo_As0);
      JsonTypeInfo.As jsonTypeInfo_As1 = asPropertyTypeDeserializer0.getTypeInclusion();
      assertSame(jsonTypeInfo_As0, jsonTypeInfo_As1);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer((JavaType) null, (TypeIdResolver) null, "JSON", true, (JavaType) null);
      PropertyName propertyName0 = new PropertyName("");
      Class<InputStream> class0 = InputStream.class;
      VirtualAnnotatedMember virtualAnnotatedMember0 = new VirtualAnnotatedMember((TypeResolutionContext) null, class0, "JSON", (JavaType) null);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED;
      BeanProperty.Std beanProperty_Std0 = new BeanProperty.Std(propertyName0, (JavaType) null, propertyName0, (Annotations) null, virtualAnnotatedMember0, propertyMetadata0);
      TypeDeserializer typeDeserializer0 = asPropertyTypeDeserializer0.forProperty(beanProperty_Std0);
      assertNotSame(typeDeserializer0, asPropertyTypeDeserializer0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(javaType0, (TypeFactory) null);
      JsonTypeInfo.As jsonTypeInfo_As0 = JsonTypeInfo.As.WRAPPER_OBJECT;
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(javaType0, classNameIdResolver0, "", false, javaType0, jsonTypeInfo_As0);
      TypeDeserializer typeDeserializer0 = asPropertyTypeDeserializer0.forProperty((BeanProperty) null);
      assertSame(typeDeserializer0, asPropertyTypeDeserializer0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonParser jsonParser0 = jsonFactory0.createParser("JSON");
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer((JavaType) null, (TypeIdResolver) null, "JSON", false, (JavaType) null);
      Object object0 = asPropertyTypeDeserializer0.deserializeTypedFromAny(jsonParser0, defaultDeserializationContext_Impl0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonParser jsonParser0 = jsonFactory0.createParser("JSON");
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer((JavaType) null, (TypeIdResolver) null, "JSON", false, (JavaType) null);
      TokenBuffer tokenBuffer0 = new TokenBuffer(jsonParser0);
      Object object0 = asPropertyTypeDeserializer0._deserializeTypedUsingDefaultImpl(jsonParser0, defaultDeserializationContext_Impl0, tokenBuffer0);
      assertNull(object0);
  }
}