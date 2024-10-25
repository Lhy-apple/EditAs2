/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:15:59 GMT 2023
 */

package org.jsoup.helper;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PipedInputStream;
import java.net.MalformedURLException;
import java.net.Proxy;
import java.net.URL;
import java.net.UnknownServiceException;
import java.nio.charset.IllegalCharsetNameException;
import java.util.Collection;
import java.util.ConcurrentModificationException;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import javax.net.ssl.SSLSocketFactory;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.net.MockURL;
import org.evosuite.runtime.testdata.EvoSuiteURL;
import org.evosuite.runtime.testdata.NetworkHandling;
import org.jsoup.Connection;
import org.jsoup.helper.HttpConnection;
import org.jsoup.parser.Parser;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class HttpConnection_ESTest extends HttpConnection_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      Connection connection0 = httpConnection0.followRedirects(false);
      assertSame(connection0, httpConnection0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      URL uRL0 = MockURL.getHttpExample();
      Connection connection0 = httpConnection0.url(uRL0);
      EvoSuiteURL evoSuiteURL0 = new EvoSuiteURL("http://www.someFakeButWellFormedURL.org/fooExample");
      NetworkHandling.createRemoteTextFile(evoSuiteURL0, "application/x-www-form-urlencoded");
      connection0.get();
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      Connection connection0 = httpConnection0.sslSocketFactory((SSLSocketFactory) null);
      assertSame(httpConnection0, connection0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      HttpConnection.Request httpConnection_Request0 = new HttpConnection.Request();
      boolean boolean0 = httpConnection_Request0.ignoreContentType();
      assertFalse(boolean0);
      assertFalse(httpConnection_Request0.ignoreHttpErrors());
      assertEquals("UTF-8", httpConnection_Request0.postDataCharset());
      assertEquals(30000, httpConnection_Request0.timeout());
      assertEquals(1048576, httpConnection_Request0.maxBodySize());
      assertTrue(httpConnection_Request0.followRedirects());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      Connection connection0 = httpConnection0.parser((Parser) null);
      assertSame(connection0, httpConnection0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      Connection connection0 = httpConnection0.ignoreContentType(false);
      assertSame(httpConnection0, connection0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      Proxy proxy0 = Proxy.NO_PROXY;
      Connection connection0 = httpConnection0.proxy(proxy0);
      assertSame(connection0, httpConnection0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      Connection.Request connection_Request0 = httpConnection0.request();
      boolean boolean0 = connection_Request0.followRedirects();
      assertTrue(boolean0);
      assertFalse(connection_Request0.ignoreHttpErrors());
      assertEquals(1048576, connection_Request0.maxBodySize());
      assertEquals("UTF-8", connection_Request0.postDataCharset());
      assertFalse(connection_Request0.ignoreContentType());
      assertEquals(30000, connection_Request0.timeout());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      Connection.Response connection_Response0 = httpConnection_Response0.removeCookie("kl7H9aq8Uj");
      assertSame(connection_Response0, httpConnection_Response0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      HttpConnection httpConnection0 = new HttpConnection();
      Connection connection0 = httpConnection0.response((Connection.Response) httpConnection_Response0);
      assertSame(connection0, httpConnection0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      // Undeclared exception!
      try { 
        httpConnection0.maxBodySize((-1196));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // maxSize must be 0 (unlimited) or larger
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      Connection connection0 = httpConnection0.userAgent("Content-Type");
      assertSame(connection0, httpConnection0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      // Undeclared exception!
      try { 
        httpConnection0.postDataCharset("Content-Encoding");
        fail("Expecting exception: IllegalCharsetNameException");
      
      } catch(IllegalCharsetNameException e) {
         //
         // Content-Encoding
         //
         verifyException("org.jsoup.helper.HttpConnection$Request", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      Connection.Response connection_Response0 = httpConnection0.response();
      assertNull(connection_Response0.charset());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      Connection connection0 = httpConnection0.timeout(2);
      assertSame(connection0, httpConnection0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      PipedInputStream pipedInputStream0 = new PipedInputStream(2);
      Connection connection0 = httpConnection0.data("org.jsoup.parser.Token$1", "org.jsoup.parser.Token$1", (InputStream) pipedInputStream0);
      assertSame(connection0, httpConnection0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      Connection.Method connection_Method0 = Connection.Method.HEAD;
      Connection connection0 = httpConnection0.method(connection_Method0);
      assertSame(connection0, httpConnection0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      // Undeclared exception!
      try { 
        HttpConnection.connect("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Malformed URL: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36
         //
         verifyException("org.jsoup.helper.HttpConnection", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      URL uRL0 = MockURL.getFtpExample();
      URL uRL1 = HttpConnection.encodeUrl(uRL0);
      assertNotSame(uRL1, uRL0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      Connection.Request connection_Request0 = httpConnection0.request();
      httpConnection0.request(connection_Request0);
      assertEquals(1048576, connection_Request0.maxBodySize());
      assertFalse(connection_Request0.ignoreContentType());
      assertEquals("UTF-8", connection_Request0.postDataCharset());
      assertFalse(connection_Request0.ignoreHttpErrors());
      assertTrue(connection_Request0.followRedirects());
      assertEquals(30000, connection_Request0.timeout());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      Connection connection0 = httpConnection0.referrer("~f8");
      assertSame(connection0, httpConnection0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      URL uRL0 = MockURL.getHttpExample();
      Connection connection0 = HttpConnection.connect(uRL0);
      assertNotNull(connection0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      byte[] byteArray0 = new byte[0];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      httpConnection0.data("Content-Type", "Content-Type", (InputStream) byteArrayInputStream0, "Content-Type");
      URL uRL0 = MockURL.getHttpExample();
      httpConnection0.url(uRL0);
      try { 
        httpConnection0.post();
        fail("Expecting exception: UnknownServiceException");
      
      } catch(UnknownServiceException e) {
         //
         // protocol doesn't support output
         //
         verifyException("org.evosuite.runtime.mock.java.net.EvoHttpURLConnection", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      URL uRL0 = MockURL.getHttpExample();
      httpConnection0.url(uRL0);
      EvoSuiteURL evoSuiteURL0 = new EvoSuiteURL("http://www.someFakeButWellFormedURL.org/fooExample");
      NetworkHandling.createRemoteTextFile(evoSuiteURL0, "application/x-www-form-urlencoded");
      Connection.Request connection_Request0 = httpConnection0.request();
      HttpConnection.Response httpConnection_Response0 = HttpConnection.Response.execute(connection_Request0);
      HttpConnection.Response httpConnection_Response1 = httpConnection_Response0.charset("\"nwN~omgs$Axa");
      // Undeclared exception!
      try { 
        httpConnection_Response1.body();
        fail("Expecting exception: IllegalCharsetNameException");
      
      } catch(IllegalCharsetNameException e) {
         //
         // \"nwN~omgs$Axa
         //
         verifyException("java.nio.charset.Charset", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      // Undeclared exception!
      try { 
        httpConnection_Response0.bufferUp();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Request must be executed (with .execute(), .get(), or .post() before getting response body
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      // Undeclared exception!
      try { 
        httpConnection_Response0.bodyStream();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Request must be executed (with .execute(), .get(), or .post() before getting response body
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      String string0 = httpConnection_Response0.statusMessage();
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      // Undeclared exception!
      try { 
        httpConnection_Response0.bodyAsBytes();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Request must be executed (with .execute(), .get(), or .post() before getting response body
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      String string0 = httpConnection_Response0.charset();
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      int int0 = httpConnection_Response0.statusCode();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      HttpConnection.KeyVal httpConnection_KeyVal0 = HttpConnection.KeyVal.create("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36", "application/x-www-form-urlencoded");
      httpConnection_KeyVal0.inputStream();
      assertEquals("application/x-www-form-urlencoded", httpConnection_KeyVal0.value());
      assertEquals("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36", httpConnection_KeyVal0.key());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream(510);
      HttpConnection.KeyVal httpConnection_KeyVal0 = HttpConnection.KeyVal.create("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36", (InputStream) pipedInputStream0);
      String string0 = httpConnection_KeyVal0.contentType();
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      HttpConnection.KeyVal httpConnection_KeyVal0 = HttpConnection.KeyVal.create("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36", "application/x-www-form-urlencoded");
      String string0 = httpConnection_KeyVal0.toString();
      assertEquals("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36=application/x-www-form-urlencoded", string0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      HashMap<String, String> hashMap0 = new HashMap<String, String>();
      hashMap0.put("Content-Type", "Content-Encoding");
      Connection connection0 = httpConnection0.data((Map<String, String>) hashMap0);
      assertSame(connection0, httpConnection0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      String[] stringArray0 = new String[1];
      // Undeclared exception!
      try { 
        httpConnection0.data(stringArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Must supply an even number of key value pairs
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      String[] stringArray0 = new String[0];
      Connection connection0 = httpConnection0.data(stringArray0);
      assertSame(httpConnection0, connection0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      String[] stringArray0 = new String[6];
      // Undeclared exception!
      try { 
        httpConnection0.data(stringArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Data key must not be empty
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      LinkedHashSet<Connection.KeyVal> linkedHashSet0 = new LinkedHashSet<Connection.KeyVal>();
      Connection connection0 = httpConnection0.data((Collection<Connection.KeyVal>) linkedHashSet0);
      assertSame(connection0, httpConnection0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      LinkedHashSet<Connection.KeyVal> linkedHashSet0 = new LinkedHashSet<Connection.KeyVal>();
      linkedHashSet0.add((Connection.KeyVal) null);
      // Undeclared exception!
      try { 
        httpConnection0.data((Collection<Connection.KeyVal>) linkedHashSet0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Key val must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      httpConnection0.data("application/x-www-form-urlencoded", "FoyrZ2q^");
      Connection.KeyVal connection_KeyVal0 = httpConnection0.data("Content-Type");
      assertNull(connection_KeyVal0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      byte[] byteArray0 = new byte[3];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      httpConnection0.data("=Kyau,E'}JqJof?$X", "Content-Encoding", (InputStream) byteArrayInputStream0, "=Kyau,E'}JqJof?$X");
      Connection.KeyVal connection_KeyVal0 = httpConnection0.data("=Kyau,E'}JqJof?$X");
      assertEquals("=Kyau,E'}JqJof?$X", connection_KeyVal0.contentType());
      assertEquals("Content-Encoding", connection_KeyVal0.value());
      assertNotNull(connection_KeyVal0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      HashMap<String, String> hashMap0 = new HashMap<String, String>();
      hashMap0.put("=ht ,$mzri-aw~C", "application/x-www-form-urlencoded");
      Connection connection0 = httpConnection0.headers(hashMap0);
      assertSame(connection0, httpConnection0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      URL uRL0 = MockURL.getHttpExample();
      Connection connection0 = httpConnection0.url(uRL0);
      Connection connection1 = connection0.header("Content-Type", "Content-Encoding");
      try { 
        connection1.post();
        fail("Expecting exception: UnknownServiceException");
      
      } catch(UnknownServiceException e) {
         //
         // protocol doesn't support output
         //
         verifyException("org.evosuite.runtime.mock.java.net.EvoHttpURLConnection", e);
      }
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      Connection connection0 = httpConnection0.header("Content-Type", (String) null);
      assertSame(httpConnection0, connection0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      httpConnection_Response0.header("Content-Encoding", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36");
      Map<String, List<String>> map0 = httpConnection_Response0.headers;
      // Undeclared exception!
      try { 
        httpConnection_Response0.processResponseHeaders(map0);
        fail("Expecting exception: ConcurrentModificationException");
      
      } catch(ConcurrentModificationException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.ArrayList$Itr", e);
      }
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      URL uRL0 = MockURL.getHttpExample();
      httpConnection0.url(uRL0);
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      httpConnection_Response0.cookie("Content-Type", "application/x-www-form-urlencoded");
      Connection.Request connection_Request0 = httpConnection0.request();
      try { 
        HttpConnection.Response.execute(connection_Request0, httpConnection_Response0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // HTTP error fetching URL
         //
         verifyException("org.jsoup.helper.HttpConnection$Response", e);
      }
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      HttpConnection.Request httpConnection_Request0 = new HttpConnection.Request();
      // Undeclared exception!
      try { 
        httpConnection_Request0.timeout((-177));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Timeout milliseconds must be 0 (infinite) or greater
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      Connection.Request connection_Request0 = httpConnection0.request();
      connection_Request0.maxBodySize(919);
      assertEquals(919, connection_Request0.maxBodySize());
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      Connection connection0 = httpConnection0.postDataCharset("iso-8859-1");
      assertSame(connection0, httpConnection0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      URL uRL0 = MockURL.getFileExample();
      httpConnection0.url(uRL0);
      try { 
        httpConnection0.get();
        fail("Expecting exception: MalformedURLException");
      
      } catch(MalformedURLException e) {
         //
         // Only http & https protocols supported
         //
         verifyException("org.jsoup.helper.HttpConnection$Response", e);
      }
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      URL uRL0 = MockURL.getHttpExample();
      httpConnection0.url(uRL0);
      Connection connection0 = httpConnection0.data("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36");
      try { 
        connection0.post();
        fail("Expecting exception: UnknownServiceException");
      
      } catch(UnknownServiceException e) {
         //
         // protocol doesn't support output
         //
         verifyException("org.evosuite.runtime.mock.java.net.EvoHttpURLConnection", e);
      }
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      URL uRL0 = MockURL.getHttpExample();
      httpConnection0.requestBody("kJe$EUGdG(WR(%.^");
      Connection connection0 = httpConnection0.url(uRL0);
      connection0.data("Content-Type", "multipart/form-data");
      try { 
        httpConnection0.post();
        fail("Expecting exception: UnknownServiceException");
      
      } catch(UnknownServiceException e) {
         //
         // protocol doesn't support output
         //
         verifyException("org.evosuite.runtime.mock.java.net.EvoHttpURLConnection", e);
      }
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      URL uRL0 = MockURL.getHttpExample();
      httpConnection0.ignoreHttpErrors(true);
      Connection connection0 = httpConnection0.url(uRL0);
      try { 
        connection0.get();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Could not find: www.someFakeButWellFormedURL.org
         //
         verifyException("org.evosuite.runtime.mock.java.net.EvoHttpURLConnection", e);
      }
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      URL uRL0 = MockURL.getHttpExample();
      httpConnection0.url(uRL0);
      EvoSuiteURL evoSuiteURL0 = new EvoSuiteURL("http://www.someFakeButWellFormedURL.org/fooExample");
      NetworkHandling.createRemoteTextFile(evoSuiteURL0, "application/x-www-form-urlencoded");
      Connection.Request connection_Request0 = httpConnection0.request();
      HttpConnection.Response httpConnection_Response0 = HttpConnection.Response.execute(connection_Request0);
      httpConnection_Response0.body();
      httpConnection_Response0.parse();
      assertEquals(30000, connection_Request0.timeout());
      assertEquals("UTF-8", connection_Request0.postDataCharset());
      assertFalse(connection_Request0.ignoreContentType());
      assertTrue(connection_Request0.followRedirects());
      assertFalse(connection_Request0.ignoreHttpErrors());
      assertEquals(1048576, connection_Request0.maxBodySize());
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      URL uRL0 = MockURL.getHttpExample();
      httpConnection0.url(uRL0);
      EvoSuiteURL evoSuiteURL0 = new EvoSuiteURL("http://www.someFakeButWellFormedURL.org/fooExample");
      NetworkHandling.createRemoteTextFile(evoSuiteURL0, "application/x-www-form-urlencoded");
      Connection.Request connection_Request0 = httpConnection0.request();
      HttpConnection.Response httpConnection_Response0 = HttpConnection.Response.execute(connection_Request0);
      httpConnection_Response0.body();
      httpConnection_Response0.body();
      assertFalse(connection_Request0.ignoreHttpErrors());
      assertEquals(30000, connection_Request0.timeout());
      assertEquals(1048576, connection_Request0.maxBodySize());
      assertFalse(connection_Request0.ignoreContentType());
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      URL uRL0 = MockURL.getHttpExample();
      Connection connection0 = httpConnection0.url(uRL0);
      connection0.proxy("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36", 160);
      try { 
        httpConnection0.post();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // java.lang.UnsupportedOperationException: Method not implemented.
         //
         verifyException("org.evosuite.runtime.mock.java.net.MockURL", e);
      }
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      URL uRL0 = MockURL.getHttpExample();
      httpConnection0.url(uRL0);
      HashMap<String, String> hashMap0 = new HashMap<String, String>();
      hashMap0.put("ismap", "x_#i ajvwqj-q(o=");
      httpConnection0.cookies(hashMap0);
      try { 
        httpConnection0.post();
        fail("Expecting exception: UnknownServiceException");
      
      } catch(UnknownServiceException e) {
         //
         // protocol doesn't support output
         //
         verifyException("org.evosuite.runtime.mock.java.net.EvoHttpURLConnection", e);
      }
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      LinkedList<String> linkedList0 = new LinkedList<String>();
      HashMap<String, List<String>> hashMap0 = new HashMap<String, List<String>>();
      hashMap0.put((String) null, linkedList0);
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      httpConnection_Response0.processResponseHeaders(hashMap0);
      assertFalse(hashMap0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      LinkedList<String> linkedList0 = new LinkedList<String>();
      linkedList0.add("%20");
      HashMap<String, List<String>> hashMap0 = new HashMap<String, List<String>>();
      hashMap0.put("set-cookie", linkedList0);
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      httpConnection_Response0.processResponseHeaders(hashMap0);
      assertFalse(hashMap0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      LinkedList<String> linkedList0 = new LinkedList<String>();
      linkedList0.add((String) null);
      HashMap<String, List<String>> hashMap0 = new HashMap<String, List<String>>();
      hashMap0.put("set-cookie", linkedList0);
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      httpConnection_Response0.processResponseHeaders(hashMap0);
      assertEquals(1, hashMap0.size());
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      LinkedList<String> linkedList0 = new LinkedList<String>();
      HashMap<String, List<String>> hashMap0 = new HashMap<String, List<String>>();
      linkedList0.addLast("=|k~526yHty>}");
      hashMap0.put("set-cookie", linkedList0);
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      httpConnection_Response0.processResponseHeaders(hashMap0);
      assertNull(httpConnection_Response0.contentType());
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      URL uRL0 = MockURL.getHttpExample();
      httpConnection0.url(uRL0);
      httpConnection0.cookie("Do@9dz1E", "*<@VO");
      httpConnection0.cookie("\"/~b", "vCQ=?");
      try { 
        httpConnection0.post();
        fail("Expecting exception: UnknownServiceException");
      
      } catch(UnknownServiceException e) {
         //
         // protocol doesn't support output
         //
         verifyException("org.evosuite.runtime.mock.java.net.EvoHttpURLConnection", e);
      }
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      httpConnection0.data("Content-Type", "Content-Type");
      URL uRL0 = MockURL.getHttpExample();
      Connection connection0 = httpConnection0.url(uRL0);
      connection0.data("\r\n\r\n", "multipart/form-data");
      try { 
        httpConnection0.get();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // HTTP error fetching URL
         //
         verifyException("org.jsoup.helper.HttpConnection$Response", e);
      }
  }
}
